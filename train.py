import os
from easydict import EasyDict
import torch
import yaml
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import numpy as np
from torch.amp import autocast, GradScaler

from models.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from models.BrownianBridge.base.modules.diffusionmodules.openaimodel import convert_norm_layers_to_fp32
from models.BrownianBridge.base.modules.image_degradation.utils_image import calculate_ssim, ssim
from utils.bb_utils import weights_init
from utils.nusc_dataloader import NuscData
import logging 

logger = logging.getLogger("train_debug")
logger.setLevel(logging.DEBUG)
def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))
    
def train_cbbdm(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    n_epochs: int,
    device: torch.device,
    log_dir: str = ".",
    ema_model=None,
    scheduler=None,
    val_interval: int = 1000,
    save_interval: int = 5000,
    max_steps: int = None,
    clip_grad_norm: float = None, 
    resume_from: str = None, # resume from a checkpoint
    use_fp16: bool = False, # use 16 bit floating point precision
):
    step = 0

    # set the paths for saving results.
    checkpoint_path = f"{log_dir}/checkpoints/"
    log_path =  f"{log_dir}/logs/"
    val_recon_path = f"{log_dir}/validation/"
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(val_recon_path, exist_ok=True)
    # lists to hold the losses during trianing.
    train_losses = []
    val_losses = []

    # if resuming from a previous checkpoint
    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint.get("step", 0)
        if scheduler and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Checkpoint loaded at step {step}")

    # used to explicitly raise an error with a stack trace if there's a problem with gradients.
    torch.autograd.set_detect_anomaly(True)

    # iterate through epochs. 
    for epoch in range(n_epochs):
        logger.info(f"Epoch {epoch + 1}/{n_epochs}")
        model.train()
        for batch in tqdm(train_loader):
            # reset the gradients
            optimizer.zero_grad()
            step += 1
            # send the noisy and clean lidar range images to the device. 
            x_noisy = batch["noisy_lidar_range"].to(device)
            x_clean = batch["clean_lidar_range"].to(device)
            # Need to do this because the radar encoder takes 3 sets of tensors as input
            x_radar = [item.to(device) for item in batch['radar_vox']]

            if use_fp16:
                x_noisy.half()
                x_clean.half()
                x_radar = [item.half() for item in x_radar]

            # compute the loss of the model. 
            loss, log = model(x_noisy, x_clean, context=x_radar)
            # backpropagation
            loss.backward()
            # step the optimizer.
            optimizer.step()

            # if using learning scheduler. 
            if scheduler is not None:
                scheduler.step(loss.item())
            train_losses.append((step, loss.item())) 

            if step % 100 == 0 or step == 1:
                logger.info(f"Step {step} | Loss: {loss:.4f}")

            # perform validation
            if step % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_batch = next(iter(val_loader))

                    val_noisy = val_batch["noisy_lidar_range"].to(device)
                    val_clean = val_batch["clean_lidar_range"].to(device)
                    # Need to do this because the radar encoder takes 3 sets of tensors as input
                    val_radar = [item.to(device) for item in val_batch['radar_vox']]
                    if use_fp16:
                        val_clean.half()
                        val_noisy.half()
                        val_radar = [item.half() for item in val_radar]
                    val_loss, _ = model(val_noisy, val_clean, context=val_radar)
                    val_losses.append((step, loss.item()))
                    logger.info(f"[VAL @ step {step}] Loss: {val_loss:.4f}")

                    if step % 2000 == 0:
                        val_recon = model.sample(val_noisy, val_radar)
                        np.save(f"{val_recon_path}validation_epoch_{epoch}_step_{step}.npy", val_recon.cpu().numpy())
                        np.save(f"{val_recon_path}gt_epoch_{epoch}_step_{step}.npy", val_clean.cpu().numpy())
                model.train()

            if step % save_interval == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }
                if scheduler:
                    ckpt["scheduler"] = scheduler.state_dict()

                ckpt_path = os.path.join(checkpoint_path, f"ckpt_step_{step}.pth")
                torch.save(ckpt, ckpt_path)
                logger.info(f"Saved checkpoint at step {step}")
                np.save(f"{log_dir}/train_losses_step_{step}.npy", np.array(train_losses))
                np.save(f"{log_dir}/val_losses_step_{step}.npy", np.array(val_losses))

            if max_steps and step >= max_steps:
                logger.warning("Reached max training steps.")
                break


def main():
    # TODO: Have better argument handling than this
    config_path = sys.argv[1]
    log_path = sys.argv[2]
    log_name = sys.argv[3]
    checkpoint_path = sys.argv[4] if len(sys.argv) == 5 else None

    # Prevent duplicate handlers if this gets run more than once (e.g., in notebook/testing)
    if not logger.handlers:
        fh = logging.FileHandler(f"{log_name}.log", mode='a')
        fh.setLevel(logging.DEBUG)

        # Add a formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        logger.addHandler(fh)
        logger.propagate = False  # Don't pass logs to root logger
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nusc = NuScenes(version='v1.0-trainval', dataroot=config.data.clean_data_path, verbose=True)

    train_dataset =  NuscData(nusc, config.data.noisy_data_path, is_train=True, nsweeps=1)
    val_dataset =  NuscData(nusc, config.data.noisy_data_path, is_train=False, nsweeps=1)
    train_loader = DataLoader(train_dataset, batch_size=config.data.train.batch_size,
                              shuffle=config.data.train.shuffle, num_workers=1, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.data.val.batch_size,
                            shuffle=config.data.val.shuffle, num_workers=0, drop_last=True)

    model = LatentBrownianBridgeModel(config.model, device).to(device)
    model.denoise_fn.apply(weights_init)
    if config.model.BB.params.UNetParams.use_fp16:
        model = model.half()
        convert_norm_layers_to_fp32(model)
   
    ema = None
    optim_config = config.model.BB.optimizer
    if optim_config.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.get_parameters(),
            lr=optim_config.lr,
            betas=(optim_config.beta1, 0.999),
            weight_decay=optim_config.weight_decay
        )
    else:
        raise NotImplementedError("Only Adam optimizer is supported right now")

    sched_config = config.model.BB.lr_scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=sched_config.factor,
        patience=sched_config.patience,
        threshold=sched_config.threshold,
        cooldown=sched_config.cooldown,
        min_lr=sched_config.min_lr
    )
    train_cbbdm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=config.training.n_epochs,
        device=device,
        log_dir=f"{log_path}/{config.model.model_name}",
        ema_model=ema,
        scheduler=scheduler,
        val_interval=config.training.val_interval,
        save_interval=config.training.save_interval,
        max_steps=config.training.n_steps,
        resume_from=checkpoint_path
    )

import os
main()