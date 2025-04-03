import os
import torch
import yaml
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes


from models.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from utils.nusc_dataloader import NuscData

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
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
    sample_fn=None,
    max_steps: int = None,
    clip_grad_norm: float = None,
):
    step = 0

    checkpoint_path = f"{log_dir}/checkpoints/"
    log_path =  f"{log_dir}/logs/"
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)


    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        model.train()

        for batch in tqdm(train_loader):
            step += 1

            x = batch["target"].to(device)
            x_cond = batch["input"].to(device)
            radar = batch["radar"].to(device)

            loss, log = model(x, x_cond, context=radar)

            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            if ema_model is not None:
                ema_model(model)

            if step % 100 == 0 or step == 1:
                print(f"Step {step} | Loss: {log['loss'].item():.4f}")

            if step % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_batch = next(iter(val_loader))
                    val_x = val_batch["target"].to(device)
                    val_cond = val_batch["input"].to(device)
                    val_radar = val_batch["radar"].to(device)

                    val_loss, _ = model(val_x, val_cond, context=val_radar)
                    print(f"[VAL @ step {step}] Loss: {val_loss.item():.4f}")
                model.train()

            if sample_fn is not None and step % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    sample_fn(model, step, save_dir=log_dir)
                model.train()

            if step % save_interval == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }
                if scheduler:
                    ckpt["scheduler"] = scheduler.state_dict()
                if ema_model:
                    ckpt["ema"] = ema_model.state_dict()

                ckpt_path = os.path.join(checkpoint_path, f"ckpt_step_{step}.pth")
                torch.save(ckpt, ckpt_path)
                print(f"Saved checkpoint at step {step}")

            if max_steps and step >= max_steps:
                print("Reached max training steps.")
                return


def main():
    # TODO: Have better argument handling than this
    config_path = sys.argv[1]
    data_path = sys.argv[2]
    log_path = sys.argv[3] 
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nusc = NuScenes(version='v1.0-mini', dataroot=data_path, verbose=True)

    train_dataset =  NuscData(nusc, is_train=True, nsweeps=1)
    val_dataset =  NuscData(nusc, is_train=False, nsweeps=1)
    train_loader = DataLoader(train_dataset, batch_size=config['data']['train']['batch_size'],
                              shuffle=config['data']['train']['shuffle'], num_workers=8, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['val']['batch_size'],
                            shuffle=config['data']['val']['shuffle'], num_workers=8, drop_last=True)

    model = LatentBrownianBridgeModel(config['model']).to(device)
    ema = None
    optim_config = config['model']['BB']['optimizer']
    if optim_config['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(model.get_parameters(), lr=optim_config['lr'], betas=(optim_config['beta1'], 0.999),
                                     weight_decay=optim_config['weight_decay'])
    else:
        raise NotImplementedError("Only Adam optimizer is supported right now")
    
    sched_config = config['model']['BB']['lr_scheduler']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=sched_config['factor'],
                                                           patience=sched_config['patience'],
                                                           threshold=sched_config['threshold'],
                                                           cooldown=sched_config['cooldown'],
                                                           min_lr=sched_config['min_lr'])

    train_cbbdm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=config['training']['n_epochs'],
        device=device,
        log_path=f"{log_path}{config['model']['model_name']}",
        ema_model=ema,
        scheduler=scheduler,
        val_interval=config['training']['val_interval'],
        save_interval=config['training']['save_interval'],
        max_steps=config['training']['n_steps']
    )
