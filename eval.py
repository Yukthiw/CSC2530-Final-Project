import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
from easydict import EasyDict
import yaml
from models.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from utils.lidar_encoder_utils import Lidar_to_range_image
from utils.nusc_dataloader import NuscData
from models.BrownianBridge.base.modules.diffusionmodules.openaimodel import convert_norm_layers_to_fp32
from utils.weather_augmentation.analyze_pc import AnalyzePointCloud
import torch.nn.functional as F

@torch.no_grad()
def remove_zero_padding(pc, threshold=0):
    return pc[torch.norm(pc, dim=-1) > threshold]

def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))
    
@torch.no_grad()
def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Computes PSNR between predicted and target images.
    
    Args:
        pred: predicted image (B,C,H,W)
        target: target image (B,C,H,W)
        max_val: max value in image (1.0 if normalized)
    
    Returns:
        PSNR value in dB, shape (B,)
    """
    mse = F.mse_loss(pred, target, reduction='none')
    mse_per_sample = mse.view(mse.shape[0], -1).mean(dim=1)  # PSNR per batch item
    psnr = 10 * torch.log10(max_val**2 / mse_per_sample)
    return psnr

@torch.no_grad()
def evaluate(model, val_loader, device, config, save_dir=None):
    model.eval()
    mse_improved, mse_noisy = [], []
    chamfer_improved, chamfer_noisy = [], []
    psnr_improved, psnr_noisy = [], []
    l2r = Lidar_to_range_image()
    pc_metrics = AnalyzePointCloud()
    save_dest = f"{save_dir}/{config.model.model_name}/"

    os.makedirs(save_dest, exist_ok=True)

    for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
        x_noisy = batch['noisy_lidar_range'].to(device)
        x_clean = batch['clean_lidar_range'].to(device)
        x_radar = [item.to(device) for item in batch['radar_vox']]
        pcs_clean = batch['clean_lidar_pc']
        pcs_noisy = batch['noisy_lidar_pc']

        # Generate sample
        x_pred = model.sample(x_noisy, x_radar)

        # Compute image MSE
        mse_improved.append(torch.mean((x_pred - x_clean) ** 2).item())
        mse_noisy.append(torch.mean((x_noisy - x_clean) ** 2).item())
        psnr_improved.append(torch.mean(compute_psnr(x_pred, x_clean)).item())
        psnr_noisy.append(torch.mean(compute_psnr(x_noisy, x_clean)).item())

        pred_pcs = l2r.to_pc_torch(x_pred)
        
        # Convert to point clouds
        for j in range(x_pred.shape[0]):
            pred_pc_clean = l2r.remove_outliers(pred_pcs[j])
            clean_pc = remove_zero_padding(pcs_clean[j])
            noisy_pc = remove_zero_padding(pcs_noisy[j])
            chamfer_improved.append(pc_metrics.chamfer_distance(pred_pc_clean.cpu(), clean_pc.cpu()))
            chamfer_noisy.append(pc_metrics.chamfer_distance(noisy_pc.cpu(), clean_pc.cpu()))

            if j == 0 and i < 10:
                np.save(os.path.join(save_dest, f"pred_pc_{i}.npy"), pred_pc_clean.cpu().numpy())
                np.save(os.path.join(save_dest, f"clean_pc_{i}.npy"), clean_pc.cpu().numpy())
                np.save(os.path.join(save_dest, f"noisy_pc_{i}.npy"), noisy_pc.cpu().numpy())
        
        if i % 100 == 0:
            np.save(os.path.join(save_dest, f"mse_recon_step_{i}.npy"), np.array(mse_improved))
            np.save(os.path.join(save_dest, f"mse_noisy_step_{i}.npy"), np.array(mse_noisy))
            np.save(os.path.join(save_dest, f"chamfer_recon_step_{i}.npy"), np.array(chamfer_improved))
            np.save(os.path.join(save_dest, f"chamfer_noisy_step_{i}.npy"), np.array(chamfer_noisy))
            np.save(os.path.join(save_dest, f"psnr_recon_step_{i}.npy"), np.array(psnr_improved))
            np.save(os.path.join(save_dest, f"psnr_noisy_step_{i}.npy"), np.array(psnr_noisy))

            np.save(os.path.join(save_dest, f"pred_range_{i}.npy"), x_pred.cpu().numpy())
            np.save(os.path.join(save_dest, f"clean_range_{i}.npy"), x_clean.cpu().numpy())
            np.save(os.path.join(save_dest, f"noisy_range_{i}.npy"), x_noisy.cpu().numpy())

    print("\n--- Evaluation Results ---")
    print(f"Range Image MSE (Reconstructed): {np.mean(mse_improved):.6f}")
    print(f"Range Image MSE (Noisy):    {np.mean(mse_noisy):.6f}")
    print(f"Chamfer Distance (Reconstructed): {np.mean(chamfer_improved):.6f}")
    print(f"Chamfer Distance (Noisy):    {np.mean(chamfer_noisy):.6f}")
    print(f"PSNR (Reconstructed): {np.mean(psnr_improved):.6f}")
    print(f"PSNR (Noisy):    {np.mean(psnr_noisy):.6f}")

    np.save(os.path.join(save_dest, f"mse_recon_final.npy"), np.array(mse_improved))
    np.save(os.path.join(save_dest, f"mse_noisy_final.npy"), np.array(mse_noisy))
    np.save(os.path.join(save_dest, f"chamfer_recon_final.npy"), np.array(chamfer_improved))
    np.save(os.path.join(save_dest, f"chamfer_noisy_final.npy"), np.array(chamfer_noisy))
    np.save(os.path.join(save_dest, f"psnr_recon_final.npy"), np.array(psnr_improved))
    np.save(os.path.join(save_dest, f"psnr_noisy_final.npy"), np.array(psnr_noisy))

    return {
        "mse_improved": mse_improved,
        "mse_noisy": mse_noisy,
        "chamfer_improved": chamfer_improved,
        "chamfer_noisy": chamfer_noisy,
    }

def main():
    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    save_dir = sys.argv[3] if len(sys.argv) > 3 else "./eval_outputs"

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=config.data.clean_data_path, verbose=True)
    val_dataset = NuscData(nusc, config.data.noisy_data_path, is_train=False, nsweeps=1)
    val_loader = DataLoader(val_dataset, batch_size=config.data.val.batch_size, num_workers=0, shuffle=False)

    # Load model
    model = LatentBrownianBridgeModel(config.model, device).to(device)
    if config.model.BB.params.UNetParams.use_fp16:
        model = model.half()
        convert_norm_layers_to_fp32(model)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded model from {checkpoint_path}")

    # Evaluate
    metrics = evaluate(model, val_loader, device, config, save_dir=save_dir)

if __name__ == "__main__":
    main()
