import itertools
import pdb
import random
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

from models.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from models.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from models.voxelnet import VoxelNet



def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        # Loading Radar Encoder
        self.radar_encoder = VoxelNet()
        encoder_checkpoint = torch.load(model_config.RADAR_ENCODER.checkpoint_path)
        self.radar_encoder.load_state_dict(encoder_checkpoint['radar_encoder_state_dict'])
        
        self.lidar_encoder = None

    # TODO: Figure out exactly how ema works and whether we want to use it
    def get_ema_net(self):
        return self

    def get_parameters(self):
        print("get parameters to optimize: UNet")
        params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    def forward(self, x_noisy, x_clean, x_radar):
        with torch.no_grad():
            x_latent = self.encode_lidar(x_noisy)
            x_cond_latent = self.encode_lidar(x_clean)
            x_radar_latent = self.encode_radar(x_radar)

        return super().forward(x_latent.detach(), x_cond_latent.detach(), x_radar_latent.detach())

    # Don't think we need this?
    # def get_cond_stage_context(self, radar_vox):
    #     with torch.no_grad():
    #         context = self.encode_rader(radar_vox)
    #     return context

    @torch.no_grad()
    def encode_lidar(self, x, normalize=None):
        # normalize = self.model_config.normalize_latent if normalize is None else normalize
        model = self.lidar_encoder
        x_latent = model.encoder(x)

        # TODO: Figure out if we want to normalize latents?
        # if not self.model_config.latent_before_quant_conv:
        #     x_latent = model.quant_conv(x_latent)
        # if normalize:
        #     if cond:
        #         x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
        #     else:
        #         x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
        return x_latent

    @torch.no_grad()
    def decode_lidar(self, x_latent, normalize=None):
        # TODO: Figure out if we want to normalize latents?
        # normalize = self.model_config.normalize_latent if normalize is None else normalize
        # if normalize:
        #     if cond:
        #         x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
        #     else:
        #         x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
        model = self.lidar_encoder
        out = model.decode(x_latent)
        return out
    
    @torch.no_grad()
    def encode_radar(self, x):
        model = self.radar_encoder
        cond_latent = model.encoder(x)
        return cond_latent

    @torch.no_grad()
    def sample(self, x_cond, clip_denoised=False, sample_mid_step=False):
        x_cond_latent = self.encode_lidar(x_cond, cond=True)
        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(y=x_cond_latent,
                                                     context=self.get_cond_stage_context(x_cond),
                                                     clip_denoised=clip_denoised,
                                                     sample_mid_step=sample_mid_step)
            out_samples = []
            for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode_lidar(temp[i].detach(), cond=False)
                out_samples.append(out.to('cpu'))

            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps",
                          dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode_lidar(one_step_temp[i].detach(), cond=False)
                one_step_samples.append(out.to('cpu'))
            return out_samples, one_step_samples
        else:
            temp = self.p_sample_loop(y=x_cond_latent,
                                      context=self.get_cond_stage_context(x_cond),
                                      clip_denoised=clip_denoised,
                                      sample_mid_step=sample_mid_step)
            x_latent = temp
            out = self.decode_lidar(x_latent, cond=False)
            return out