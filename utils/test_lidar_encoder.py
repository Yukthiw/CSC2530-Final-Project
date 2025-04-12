import os
import sys
import torch
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

def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))

def main():
    # Command-line args
    # config_path = sys.argv[1]
    # data_path = sys.argv[2]
    # log_path = sys.argv[3]  # Not used now, but available

    # # Load config and setup device
    # config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load NuScenes
    nusc = NuScenes(version='v1.0-mini', dataroot='/u/yukthiw/files/nuscenes/mini', verbose=True)

    # Dataset and loader
    train_dataset = NuscData(nusc, is_train=True, nsweeps=1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=1,
        drop_last=True
    )

    # Print or visualize range images
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Range Image Batches")):
        print('insdie')
        range_imgs = batch['range_image'] if isinstance(batch, dict) and 'range_image' in batch else batch

        print(f"Batch {batch_idx}: Range Image Shape = {range_imgs.shape}")

        if batch_idx == 0:
            import matplotlib.pyplot as plt
            img = range_imgs[0].cpu().numpy()
            if img.ndim == 3:
                img = img[0]  # take first channel if multi-channel
            plt.imshow(img, cmap='gray')
            plt.title("Sample Range Image")
            plt.show()

        if batch_idx >= 4:
            break

if __name__ == "__main__":
    main()
