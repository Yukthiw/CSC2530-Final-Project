import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np

from models.BrownianBridge.base.modules.diffusionmodules.unet_v2 import UNetModelV2
from utils.bb_utils import extract, default
from models.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel
from models.BrownianBridge.base.modules.encoders.modules import SpatialRescaler

# imports for 3D loss function
from utils.lidar_encoder_utils import Lidar_to_range_image as LTR
from pytorch3d.loss import chamfer_distance


class BrownianBridgeModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        # model hyperparameters
        model_params = model_config.BB.params
        self.num_timesteps = model_params.num_timesteps
        self.mt_type = model_params.mt_type
        # TODO: Look at what max variance params work best here
        self.max_var = model_params.max_var if model_params.__contains__("max_var") else 1
        self.eta = model_params.eta if model_params.__contains__("eta") else 1
        self.skip_sample = model_params.skip_sample
        self.sample_type = model_params.sample_type
        self.sample_step = model_params.sample_step
        self.steps = None
        self.register_schedule()

        # loss and objective
        self.loss_type = model_params.loss_type
        self.objective = model_params.objective

        if self.loss_type == "chamfer":
            # custom loss function
            self.LTR = LTR() # lidar to range image class instance.
            self.range_to_pointcloud = self.LTR.to_pc_torch #

            if self.decode_lidar == None:
                ValueError("missing valid override of 'decode_lidar' in BrownianBridgeModel")
            if self.range_to_pointcloud == None:
                ValueError("missing valid override of 'range_to_pointcloud' in BrownianBridgeModel")
        


        # UNet
        self.image_size = model_params.UNetParams.image_size
        self.channels = model_params.UNetParams.in_channels
        self.condition_key = model_params.UNetParams.condition_key

        

        self.denoise_fn = UNetModelV2(**vars(model_params.UNetParams))

    # def chamfer_dist(self, pc1, pc2):
    #     """
    #     Computes Chamfer Distance between two point clouds.
    #     :param pc1: (B, N, 3)
    #     :param pc2: (B, M, 3)
    #     :return: scalar Chamfer distance averaged over the batch
    #     """
    #     # Compute pairwise distance
    #     B, N, _ = pc1.shape
    #     _, M, _ = pc2.shape

    #     # (B, N, 1, 3) - (B, 1, M, 3) => (B, N, M)
    #     pc1_expand = pc1.unsqueeze(2)  # (B, N, 1, 3)
    #     pc2_expand = pc2.unsqueeze(1)  # (B, 1, M, 3)
    #     dist = torch.norm(pc1_expand - pc2_expand, dim=-1)  # (B, N, M)

    #     # For each point in pc1, find nearest neighbor in pc2
    #     dist1 = dist.min(dim=2)[0]  # (B, N)
    #     # For each point in pc2, find nearest neighbor in pc1
    #     dist2 = dist.min(dim=1)[0]  # (B, M)

    #     chamfer = (dist1.mean(dim=1) + dist2.mean(dim=1)).mean()  # scalar
    #     return chamfer

    def register_schedule(self):
        '''
        Pre-computing variance and mixing coefficients here for forward diffusion process.
        '''
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        return self.denoise_fn.parameters()

    def forward(self, x, y, context):
        b, c, h, w, device, img_size_h, img_size_w, = *x.shape, x.device, *self.image_size
        assert h == img_size_h and w == img_size_w, f'height and width of image must be {self.image_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, y, context, t)

    def p_losses(self, x0, y, context, t, noise=None):
        """
        model loss
        :param x0: encoded x_ori (Clean Lidar)
        :param y: encoded y_ori (Noisy Lidar)
        :param context: Encoded radar
        :param t: timesteps
        :param noise: Standard Gaussian Noise
        :return: loss
        """
        b, c, h, w = x0.shape
        # compute random noise. 
        noise = default(noise, lambda: torch.randn_like(x0))
        x_t, objective = self.q_sample(x0, y, t, noise)
        objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
        if objective_recon.requires_grad:
            objective_recon.register_hook(lambda grad: print(f"'objective_recon'Grad norm: {grad.norm().item()}"))

        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        if x0_recon.requires_grad:
            x0_recon.register_hook(lambda grad: print(f"'x0_recon'Grad norm: {grad.norm().item()}"))

        # print("x0_recon: Requires grad?", x0_recon.requires_grad)
        # print("x0_recon: Grad fn?", x0_recon.grad_fn)

        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        elif self.loss_type == 'chamfer':
            # decode the latents of x0 and x0_recon
            range_x0 = self.decode_lidar(x0)
            if range_x0.requires_grad:
                range_x0.register_hook(lambda grad: print(f"'range_x0' Grad norm: {grad.norm().item()}"))

            range_x0_recon = self.decode_lidar(x0_recon) #TODO fix decode lidar to allow for gradient tracking. 
            if range_x0_recon.requires_grad:
                range_x0_recon.register_hook(lambda grad: print(f"'range_x0_recon' Grad norm: {grad.norm().item()}"))

            # print("range_x0: Requires grad?", range_x0.requires_grad)
            # print("range_x0: Grad fn?", range_x0.grad_fn)

            # convert to 3D point clouds
            pc_x0 = self.range_to_pointcloud(range_x0)
            if pc_x0.requires_grad:
                pc_x0.register_hook(lambda grad: print(f"'pc_x0' Grad norm: {grad.norm().item()}"))
            pc_x0_recon = self.range_to_pointcloud(range_x0_recon)
            if pc_x0_recon.requires_grad:
                pc_x0_recon.register_hook(lambda grad: print(f"'pc_x0_recon' Grad norm: {grad.norm().item()}"))

            # print("pc_x0: Requires grad?", pc_x0.requires_grad)
            # print("pc_x0: Grad fn?", pc_x0.grad_fn)

            # choose a 3D loss like Chamfer Distance # TODO try using pytorch3d instead, or torchmetrics.
            # cd_loss = self.chamfer_dist(pc_x0_recon[..., :3], pc_x0[..., :3]) # keep only the first 3 dimensions (x,y,z) and omit intensity.
            chamfer, _ = chamfer_distance(pc_x0_recon, pc_x0) # this should actually use the intensity i think.
            if chamfer.requires_grad:
                chamfer.register_hook(lambda grad: print(f"'chamfer' Grad norm: {grad.norm().item()}"))

            chamfer.detach()
            # print("chamfer: Requires grad?", chamfer.requires_grad)
            # print("chamfer: Grad fn?", chamfer.grad_fn)

            # optionally combine with the image based loss:
            recloss = (objective - objective_recon).abs().mean()
            if recloss.requires_grad:
                recloss.register_hook(lambda grad: print(f"'recloss original' Grad norm: {grad.norm().item()}"))
            scaled_chamfer = torch.log1p(chamfer)  # log(1 + x), safe and smooth
            if scaled_chamfer.requires_grad:
                scaled_chamfer.register_hook(lambda grad: print(f"'scaled_chamfer' Grad norm: {grad.norm().item()}"))
            recloss = recloss * scaled_chamfer
            if recloss.requires_grad:
                recloss.register_hook(lambda grad: print(f"'recloss' Grad norm: {grad.norm().item()}"))

            # Check for division by zero
            # if (x == 0).any():
            #     print("Zero division detected!")

            # # Check for log(0)
            # if (x <= 0).any():
            #     print("Log(0) detected!")

            # print("recloss: Requires grad?", recloss.requires_grad)
            # print("recloss: Grad fn?", recloss.grad_fn)
        else:
            raise NotImplementedError()

        
        log_dict = {
            "loss": recloss,
            "x0_recon": x0_recon
        }
        return recloss, log_dict

    def q_sample(self, x0, y, t, noise):
        '''
        This is where x_t is calculated for the forward process, we also return the objective
        from here to calculate loss, in traditional DDPM it would just be predicting the noise
        but in BBDM paper they use the "grad" loss function (See Algorithm 1 in https://arxiv.org/pdf/2205.07680).

        x0: Clean lidar gt latent
        y: Weather perturbed lidar latent
        t: Timestep
        '''
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == 'ysubx':
            objective = y - x0
        else:
            raise NotImplementedError()

        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,
            objective
        )

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        """
        For x0 (Clean Lidar) and y (Noisy Lidar), both of shape BxCxWxH, sample all timesteps
        and return list of length num_timesteps of image batches (all would be BxCxWxH). 
        
        This is the full forward diffusson process.
        """
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False):
        """
        Backward diffusion process (one time step).

        x_t: Lidar latent at timestep t
        y: Original noisy lidar latent
        context: Radar latent
        i: Sample step index (during inference time we skip sampling so not all trained timesteps are sampled)
        """
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample_loop(self, y, context, clip_denoised=True, sample_mid_step=False):
        """
        Full reverse process at inference time (this is what is called when actually 
        generating a clean lidar image at inference time).

        y: Noisy lidar encoding
        context: Radar encoding
        clip_denoised: 
        """
        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(self.steps, desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
            return img

    @torch.no_grad()
    def sample(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        return self.p_sample_loop(y, context, clip_denoised, sample_mid_step)