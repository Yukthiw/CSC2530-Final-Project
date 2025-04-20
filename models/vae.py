from diffusers.models import AutoencoderKL
import torch
import safetensors
from models.replace_models import replace_attn, replace_conv, replace_down

class Lidar_VAE():
    def __init__(self, vae_config, vae_checkpoint, device, use_fp16=False):
        config = AutoencoderKL.load_config(vae_config)
        vae = AutoencoderKL.from_config(config)
        vae_checkpoint = safetensors.torch.load_file(vae_checkpoint)
        if 'quant_conv.weight' not in vae_checkpoint:
            vae.quant_conv = torch.nn.Identity()
            vae.post_quant_conv = torch.nn.Identity()
        replace_down(vae)
        replace_conv(vae)
        if 'encoder.mid_block.attentions.0.to_q.weight' not in vae_checkpoint:
            replace_attn(vae)
        vae.load_state_dict(vae_checkpoint)
        self.vae = vae.to(device)
        if use_fp16:
            self.vae = vae.half()
        self.device = device
    
    def print_total_params(self):
        """Calculate and print the total number of parameters of the VAE model."""
        total_params = sum(p.numel() for p in self.vae.parameters())
        print(f"Parameters of VAE: {total_params / 1024. / 1024.} M")

    def encode_range_image(self, image):
        image = image.to(self.device)
        image = self.vae.encode(image).latent_dist.sample()
        image = image * self.vae.config.scaling_factor
        return image


    def decode_range_image(self, latents, output_type = "torch"):
        latents = latents / self.vae.config.scaling_factor
        # decode the image latents with the VAE
        image = self.vae.decode(latents).sample
        if output_type == "torch":
            return image
        elif output_type == "pil":
            # TODO: Need to fix this if we intend on using it
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = self.numpy_to_pil(image)
