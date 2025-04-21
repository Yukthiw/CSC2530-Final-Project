import sys

import numpy as np
from tqdm import tqdm
import yaml
from easydict import EasyDict
from utils.lidar_encoder_utils import Lidar_to_range_image
from utils.nusc_dataloader import NuscData
from utils.weather_augmentation.analyze_pc import AnalyzePointCloud
import torch
from models.benchmarking.DROR import dynamic_radius_outlier_filter

def load_config(path):
    with open(path, "r") as f:
        return EasyDict(yaml.safe_load(f))
    
def compute_psnr_np(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """
    Computes PSNR between predicted and target images (no batch dimension).

    Args:
        pred: predicted image, shape (C, H, W) or (H, W)
        target: target image, same shape as pred
        max_val: maximum possible pixel value (1.0 if normalized)

    Returns:
        PSNR value in dB (float)
    """
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')  # Identical images
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr

def evaluate(val_dataset, config, save_dir):
    mse_improved, mse_noisy = [], []
    chamfer_improved, chamfer_noisy = [], []
    psnr_improved, psnr_noisy = [], []
    ldc_improved, ldc_noisy = [], []
    dhl_improved, dhl_noisy = [], []

    l2r = Lidar_to_range_image()
    pc_metrics = AnalyzePointCloud()
    
    # parameters for DROR
    alpha = 0.33
    beta = 5.0
    k_min = 3
    sr_min = 0.001

    # loop through the dataset. 
    for i in tqdm(range(len(val_dataset)), desc="processing samples"):
        # load data from dataset.
        sample = val_dataset[i]
        clean_lidar_pc = sample["clean_lidar_pc"].numpy()
        clean_lidar_range = sample["clean_lidar_range"].numpy()
        noisy_lidar_pc = sample["noisy_lidar_pc"].numpy()
        noisy_lidar_range = sample["noisy_lidar_range"].numpy()
        # compute the predicted pointcloud
        mask = dynamic_radius_outlier_filter(noisy_lidar_pc, alpha=alpha, beta=beta, k_min=k_min, sr_min=sr_min)
        pred_pc = noisy_lidar_pc[mask]

        # generate the predicted range image from the pointcloud. 
        pred_range_image = l2r.generate_range_image(pred_pc)
        # reorder the dimensions (so that it will match with the loaded ones)
        pred_range_image = np.transpose(pred_range_image, (2, 1, 0))

        # keep only the xyz dimensions for chamfer distance
        # pred_pc = pred_pc[:, :3] #XYZ and intensity
        # noisy_lidar_pc = noisy_lidar_pc[:, :3]
        # clean_lidar_pc = clean_lidar_pc[:, :3]

        # print("shape of pred_pc: ", pred_pc.shape)
        # print("shape of clean_lidar_pc: ", clean_lidar_pc.shape)

        # compute chamfer distance
        print("computing chamfer")
        chamfer_improved.append(pc_metrics.chamfer_distance(pred_pc, clean_lidar_pc))
        chamfer_noisy.append(pc_metrics.chamfer_distance(noisy_lidar_pc, clean_lidar_pc))
        print("computing dhl")
        dhl_improved.append(pc_metrics.density_histogram_l1(pred_pc[:,0:3], clean_lidar_pc[:,0:3]))
        dhl_noisy.append(pc_metrics.density_histogram_l1(noisy_lidar_pc[:,0:3], clean_lidar_pc[:,0:3]))
        print("computing ldc")
        ldc_improved.append(pc_metrics.local_density_consistency(pred_pc[:,0:3], clean_lidar_pc[:,0:3]))
        ldc_noisy.append(pc_metrics.local_density_consistency(noisy_lidar_pc[:,0:3], clean_lidar_pc[:,0:3]))
        print("computing mse")
        mse_improved.append(np.mean((pred_range_image - clean_lidar_range) ** 2))
        mse_noisy.append(np.mean((noisy_lidar_range - clean_lidar_range) ** 2))
        print("computing psnr")
        psnr_improved.append(np.mean(compute_psnr_np(pred_range_image, clean_lidar_range)))
        psnr_noisy.append(np.mean(compute_psnr_np(noisy_lidar_range, clean_lidar_range)))

        # TODO why does yukthi save pred_pc, but not return it in his evaluate function?
        # TODO in his calculation of MSE, i'm not sure that it's calculating what he thinks it is. 
        # TODO why does he remove zero padding?
        if i > 10:
            break #TODO for testing only

    print("\n--- Evaluation Results ---")
    print(f"Range Image MSE (Reconstructed): {np.mean(mse_improved):.6f}")
    print(f"Range Image MSE (Noisy):    {np.mean(mse_noisy):.6f}")
    print(f"Chamfer Distance (Reconstructed): {np.mean(chamfer_improved):.6f}")
    print(f"Chamfer Distance (Noisy):    {np.mean(chamfer_noisy):.6f}")
    print(f"Density Histogram L1 Distance (Reconstructed): {np.mean(dhl_improved):.6f}")
    print(f"Density Histogram L1 Distance (Noisy):    {np.mean(dhl_noisy):.6f}")
    print(f"Local Density Consistency (Reconstructed): {np.mean(ldc_improved):.6f}")
    print(f"Local Density Consistency (Noisy):    {np.mean(ldc_noisy):.6f}")
    print(f"PSNR (Reconstructed): {np.mean(psnr_improved):.6f}")
    print(f"PSNR (Noisy):    {np.mean(psnr_noisy):.6f}")



def main():
    config_path = sys.argv[1]

    config = load_config(config_path)

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=config.data.clean_data_path, verbose=True)
    val_dataset = NuscData(nusc, config.data.noisy_data_path, is_train=False, nsweeps=1)

    # evaluate the model
    metrics = evaluate(val_dataset, config, save_dir = None)



if __name__ == "__main__":
    main()