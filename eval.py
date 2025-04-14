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
@torch.no_grad()
def remove_zero_padding(pc, threshold=0):
    return pc[torch.norm(pc, dim=-1) > threshold]

def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))
    
@torch.no_grad()
def evaluate(model, val_loader, device, config, save_dir=None):
    model.eval()
    mse_improved, mse_noisy = [], []
    chamfer_improved, chamfer_noisy = [], []
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

        # Convert to point clouds
        for j in range(x_pred.shape[0]):
            pred_pc = l2r.to_pc_torch(x_pred[j].unsqueeze(0).cpu())[0]
            clean_pc = remove_zero_padding(pcs_clean[j])
            noisy_pc = remove_zero_padding(pcs_noisy[j])

            chamfer_improved.append(pc_metrics.chamfer_distance(pred_pc.cpu(), clean_pc.cpu()))
            chamfer_noisy.append(pc_metrics.chamfer_distance(noisy_pc.cpu(), clean_pc.cpu()))

            if j == 0 and i < 10:
                np.save(os.path.join(save_dest, f"pred_pc_{i}.npy"), pred_pc.numpy())
                np.save(os.path.join(save_dest, f"clean_pc_{i}.npy"), clean_pc.numpy())
                np.save(os.path.join(save_dest, f"noisy_pc_{i}.npy"), noisy_pc.numpy())
        
        if i % 100 == 0:
            np.save(os.path.join(save_dest, f"mse_recon_step_{i}.npy"), np.array(mse_improved))
            np.save(os.path.join(save_dest, f"mse_noisy_step_{i}.npy"), np.array(mse_noisy))
            np.save(os.path.join(save_dest, f"chamfer_recon_step_{i}.npy"), np.array(chamfer_improved))
            np.save(os.path.join(save_dest, f"chamfer_recon_step_{i}.npy"), np.array(chamfer_noisy))

    print("\n--- Evaluation Results ---")
    print(f"Range Image MSE (Reconstructed): {np.mean(mse_improved):.6f}")
    print(f"Range Image MSE (Noisy):    {np.mean(mse_noisy):.6f}")
    print(f"Chamfer Distance (Reconstructed): {np.mean(chamfer_improved):.6f}")
    print(f"Chamfer Distance (Noisy):    {np.mean(chamfer_noisy):.6f}")

    np.save(os.path.join(save_dest, f"{config.model.model_name}_mse_recon_final.npy"), np.array(mse_improved))
    np.save(os.path.join(save_dest, f"{config.model.model_name}mse_noisy_final.npy"), np.array(mse_noisy))
    np.save(os.path.join(save_dest, f"{config.model.model_name}chamfer_recon_final.npy"), np.array(chamfer_improved))
    np.save(os.path.join(save_dest, f"{config.model.model_name}chamfer_recon_final.npy"), np.array(chamfer_noisy))

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
