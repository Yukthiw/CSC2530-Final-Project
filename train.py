import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_bbdm_with_radar_cond(
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
    os.makedirs(log_dir, exist_ok=True)

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

                ckpt_path = os.path.join(log_dir, f"ckpt_step_{step}.pth")
                torch.save(ckpt, ckpt_path)
                print(f"Saved checkpoint at step {step}")

            if max_steps and step >= max_steps:
                print("Reached max training steps.")
                return
