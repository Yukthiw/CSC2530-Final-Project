import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys

from models.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from utils.nusc_dataloader import NuscData
from utils.bb_utils import load_config
from models.BrownianBridge.base.modules.diffusionmodules.openaimodel import convert_norm_layers_to_fp32

@torch.no_grad()
def remove_zero_padding(pc, threshold=1e-3):
    return pc[torch.norm(pc, dim=-1) > threshold]

@torch.no_grad()
def evaluate(model, val_loader, device, save_dir=None, max_batches=50):
    model.eval()
    mse_improved, mse_noisy = [], []
    chamfer_improved, chamfer_noisy = [], []

    os.makedirs(save_dir, exist_ok=True)

    for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
        if i >= max_batches:
            break

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

        # Convert to point clouds
        for j in range(x_pred.shape[0]):
            # pred_pc = convert_range_image_to_point_cloud(x_pred[j].cpu().numpy())
            clean_pc = remove_zero_padding(torch.tensor(pcs_clean[j]))
            noisy_pc = remove_zero_padding(torch.tensor(pcs_noisy[j]))
            pred_pc = remove_zero_padding(torch.tensor(pred_pc))

            # chamfer_improved.append(chamfer_distance(pred_pc.unsqueeze(0), clean_pc.unsqueeze(0)).item())
            # chamfer_noisy.append(chamfer_distance(noisy_pc.unsqueeze(0), clean_pc.unsqueeze(0)).item())

            if j == 0 and i < 5:  # Save 5 sample reconstructions
                np.save(os.path.join(save_dir, f"pred_pc_{i}.npy"), pred_pc.numpy())
                np.save(os.path.join(save_dir, f"clean_pc_{i}.npy"), clean_pc.numpy())
                np.save(os.path.join(save_dir, f"noisy_pc_{i}.npy"), noisy_pc.numpy())

    print("\n--- Evaluation Results ---")
    print(f"Range Image MSE (Improved): {np.mean(mse_improved):.6f}")
    print(f"Range Image MSE (Noisy):    {np.mean(mse_noisy):.6f}")
    print(f"Chamfer Distance (Improved): {np.mean(chamfer_improved):.6f}")
    print(f"Chamfer Distance (Noisy):    {np.mean(chamfer_noisy):.6f}")

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
    val_loader = DataLoader(val_dataset, batch_size=config.data.val.batch_size, shuffle=False)

    # Load model
    model = LatentBrownianBridgeModel(config.model, device).to(device)
    if config.model.BB.params.UNetParams.use_fp16:
        model = model.half()
        convert_norm_layers_to_fp32(model)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded model from {checkpoint_path}")

    # Evaluate
    metrics = evaluate(model, val_loader, device, save_dir=save_dir)

if __name__ == "__main__":
    main()
