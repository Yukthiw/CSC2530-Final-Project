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
from utils.nusc_dataloader import NuscData
import logging 

logger = logging.getLogger("train_debug")
logger.setLevel(logging.DEBUG)

# Prevent duplicate handlers if this gets run more than once (e.g., in notebook/testing)
if not logger.handlers:
    fh = logging.FileHandler("train_debug.log", mode='a')
    fh.setLevel(logging.DEBUG)

    # Add a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.propagate = False  # Don't pass logs to root logger

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
    resume_from: str = None,
):
    step = 0

    checkpoint_path = f"{log_dir}/checkpoints/"
    log_path =  f"{log_dir}/logs/"
    val_recon_path = f"{log_dir}/validation/"
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    train_losses = []
    val_losses = []

    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint.get("step", 0)
        if scheduler and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Checkpoint loaded at step {step}")

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        model.train()

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            step += 1
            x_noisy = batch["noisy_lidar_range"].to(device).half()
            x_clean = batch["clean_lidar_range"].to(device).half()
            # Need to do this because the radar encoder takes 3 sets of tensors as input
            x_radar = [item.to(device).half() for item in batch['radar_vox']]
            logger.warning(f"Step: {step}")
            loss, log = model(x_noisy, x_clean, context=x_radar)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm or 1e10)

            optimizer.step()

            if scheduler is not None:
                scheduler.step(loss)
            train_losses.append((step, loss.item())) 

            logger.warning(f"Loss: {loss.item()}")
            logger.warning(f"Grad Norm: {grad_norm}")


            if step % 100 == 0 or step == 1:
                print(f"Step {step} | Loss: {loss:.4f}")

            if step % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_batch = next(iter(val_loader))
                    val_clean = val_batch["clean_lidar_range"].to(device).half()
                    val_noisy = val_batch["noisy_lidar_range"].to(device).half()
                    val_radar = [item.to(device).half() for item in val_batch['radar_vox']] 
                    val_loss, _ = model(val_clean, val_noisy, context=val_radar)
                    val_losses.append((step, loss.item()))
                    print(f"[VAL @ step {step}] Loss: {val_loss:.4f}")

                    if save_interval % step == 0:
                        val_recon = model.sample(val_noisy, val_radar)
                        np.save(f"{val_recon_path}validation_epoch_{epoch}_step_{step}.npy", val_recon.cpu().numpy())
                model.train()

            # if sample_fn is not None and step % val_interval == 0:
            #     model.eval()
            #     with torch.no_grad():
            #         sample_fn(model, step, save_dir=log_dir)
            #     model.train()

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
                print(f"Saved checkpoint at step {step}")
                np.save(f"{log_dir}/train_losses_step_{step}.npy", np.array(train_losses))
                np.save(f"{log_dir}/val_losses_step_{step}.npy", np.array(val_losses))

            if max_steps and step >= max_steps:
                print("Reached max training steps.")
                break


def main():
    # TODO: Have better argument handling than this
    config_path = sys.argv[1]
    log_path = sys.argv[2] 
    checkpoint_path = sys.argv[3] if len(sys.argv) == 4 else None

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nusc = NuScenes(version='v1.0-trainval', dataroot=config.data.clean_data_path, verbose=True)

    train_dataset =  NuscData(nusc, config.data.noisy_data_path, is_train=True, nsweeps=1)
    val_dataset =  NuscData(nusc, config.data.noisy_data_path, is_train=False, nsweeps=1)
    train_loader = DataLoader(train_dataset, batch_size=config.data.train.batch_size,
                              shuffle=config.data.train.shuffle, num_workers=1, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.data.val.batch_size,
                            shuffle=config.data.val.shuffle, num_workers=1, drop_last=True)

    model = LatentBrownianBridgeModel(config.model, device).to(device)
    if config.model.use_half_precision:
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