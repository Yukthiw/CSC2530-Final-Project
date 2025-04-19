import os
import sys
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
# Add the project root to sys.path to resolve imports properly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.nusc_dataloader import NuscData
from utils.lidar_encoder_utils import Lidar_to_range_image
from utils.viz import plot_point_cloud_comparison
def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))


def chamfer_distance(pcd1, pcd2,chunk_size=8000):
    """
    Compute the Chamfer distance between two point clouds.
    
    Parameters:
    pcd1 (Tensor): Point cloud 1 (N1 x 3)
    pcd2 (Tensor): Point cloud 2 (N2 x 3)
    
    Returns:
    float: Chamfer distance between point clouds
    """
   # Split the computation of distances into smaller chunks to prevent memory issues
    dist1_list = []
    dist2_list = []

    # Compute distances from pcd1 to pcd2 in chunks
    for start in range(0, pcd1.size(0), chunk_size):
        end = min(start + chunk_size, pcd1.size(0))
        dist1_batch = torch.cdist(pcd1[start:end], pcd2, p=2)  # Squared Euclidean distance
        dist1_list.append(dist1_batch)
        del dist1_batch  # Delete the batch to free memory
        torch.cuda.empty_cache()  # Clear cache after each batch

    # Compute distances from pcd2 to pcd1 in chunks
    for start in range(0, pcd2.size(0), chunk_size):
        end = min(start + chunk_size, pcd2.size(0))
        dist2_batch = torch.cdist(pcd2[start:end], pcd1, p=2)  # Squared Euclidean distance
        dist2_list.append(dist2_batch)
        del dist2_batch  # Delete the batch to free memory
        torch.cuda.empty_cache()  # Clear cache after each batch

    dist1 = torch.cat(dist1_list, dim=0)  # Concatenate all batches for pcd1 -> pcd2
    dist2 = torch.cat(dist2_list, dim=0)  # Concatenate all batches for pcd2 -> pcd1

    # Forward Chamfer Distance: for each point in pcd1, find the closest point in pcd2
    min_dist1, _ = dist1.min(dim=1)
    forward_chamfer = min_dist1.mean()

    # Backward Chamfer Distance: for each point in pcd2, find the closest point in pcd1
    min_dist2, _ = dist2.min(dim=1)
    backward_chamfer = min_dist2.mean()

    # Chamfer distance is the average of forward and backward distances
    return forward_chamfer + backward_chamfer


def main():
    # Command-line args
    config_path = sys.argv[1]
    data_path = sys.argv[2]
    weather_path =  sys.argv[3]
    log_path = sys.argv[4]  # Not used now, but available

    # # Load config and setup device
    # config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)

    # Dataset and loader
    train_dataset = NuscData(nusc,noisy_lidar_dataroot=weather_path, is_train=True, nsweeps=1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=config.data.train.shuffle,
        num_workers=1,
        drop_last=True
    )

    # Print or visualize range images
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Range Image Batches")):
        range_img = batch['clean_lidar_range']
        #print(range_imgs)
        print(f"Batch {batch_idx}: Range Image Shape = {range_img.shape}")
        lidar_enc_utils = Lidar_to_range_image()
        if batch_idx==0:
            pc_from_range = lidar_enc_utils.to_pc_torch(range_img)
            pc_from_range = lidar_enc_utils.remove_outliers(pc_from_range[0]).to(device)
            base_pc = batch['clean_lidar_pc'][0,:,:4].to(device)
            weather_pc = batch['noisy_lidar_pc'][0,:,:4].to(device)
            # Chamfer-style distance (symmetric nearest-neighbor)
            chamfer_dist = chamfer_distance(pc_from_range, base_pc)
            print(f"Chamfer_dist: {chamfer_dist}")
            chamfer_dist = chamfer_distance(weather_pc, base_pc)
            print(f"Perturbed chamfer_dist: {chamfer_dist}")
            plot_point_cloud_comparison(pc_from_range.cpu().numpy(), base_pc.cpu().numpy(), title1="Reconstructed", title2="GT")
        
        else:
            break

if __name__ == "__main__":
    main()
