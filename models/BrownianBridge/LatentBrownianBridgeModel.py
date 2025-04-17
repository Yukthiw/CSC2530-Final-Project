import itertools
import pdb
import random
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

from models.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from models.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from models.voxelnet import VoxelNet
from models.vae import Lidar_VAE


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, model_config, device):
        super().__init__(model_config)

        # Loading Radar Encoder
        self.radar_encoder = VoxelNet()
        encoder_checkpoint = torch.load(model_config.RADAR_ENCODER.checkpoint_path)
        self.radar_encoder.load_state_dict(encoder_checkpoint['radar_encoder_state_dict'])
        if model_config.BB.params.UNetParams.use_fp16:
            self.radar_encoder = self.radar_encoder.half()
        
        self.lidar_encoder = Lidar_VAE(model_config.LIDAR_ENCODER.config_path, 
                                       model_config.LIDAR_ENCODER.checkpoint_path,
                                       device,
                                       model_config.BB.params.UNetParams.use_fp16)

        # # freeze the weights of the lidar encoder to prevent gradient update. 
        # for param in self.lidar_encoder.vae.parameters():
        #     param.requires_grad = False

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

    def forward(self, x_noisy, x_clean, context):
        with torch.no_grad():
            x_noisy_latent = self.encode_lidar(x_noisy)
            x_clean_latent = self.encode_lidar(x_clean)
            x_radar_latent = self.encode_radar(context)

        return super().forward(x_clean_latent.detach(), x_noisy_latent.detach(), x_radar_latent.detach())

    @torch.no_grad()
    def encode_lidar(self, x, normalize=None):
        # normalize = self.model_config.normalize_latent if normalize is None else normalize
        model = self.lidar_encoder
        x_latent = model.encode_range_image(x)
        return x_latent

    @torch.no_grad() # need to get rid of this for the loss function that uses the point cloud version. Freeze the weights elsewhere. 
    def decode_lidar(self, x_latent):
        model = self.lidar_encoder
        out = model.decode_range_image(x_latent)
        return out
    
    @torch.no_grad()
    def encode_radar(self, x):
        model = self.radar_encoder
        # Radar data is a tuple which is unpacked and passed to encoder
        cond_latent = model(*x)
        return cond_latent

    @torch.no_grad()
    def sample(self, x_cond, x_radar, clip_denoised=False, sample_mid_step=False):
        x_cond_latent = self.encode_lidar(x_cond)
        x_radar_latent = self.encode_radar(x_radar)
        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(y=x_cond_latent,
                                                     context=x_radar_latent,
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
            temp = self.p_sample_loop(y=x_cond_latent.detach(),
                                      context=x_radar_latent.detach(),
                                      clip_denoised=clip_denoised,
                                      sample_mid_step=sample_mid_step)
            x_latent = temp
            out = self.decode_lidar(x_latent)
            return out