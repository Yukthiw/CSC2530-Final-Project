from nuscenes.nuscenes import NuScenes
from utils.nusc_dataloader import NuscData
from models.vae import Lidar_VAE
from torch.utils.data import DataLoader

nusc = NuScenes(version='v1.0-trainval', dataroot="/w/247/yukthiw/nuscenes_data/nuscenes/", verbose=True)
train_dataset = NuscData(nusc, noisy_lidar_dataroot="/w/246/willdormer/projects/CompImaging/augmented_pointclouds/samples/LIDAR_TOP/", is_train=True, nsweeps=1)
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=1,
    drop_last=True
)


vae = Lidar_VAE("/w/331/abarroso/vae_model/nuscenes/vae/config.json", "/w/331/abarroso/vae_model/nuscenes/vae/diffusion_pytorch_model.safetensors", "cuda")
sample = train_dataset[0]['clean_lidar_range']
for batch in train_loader:
    x = vae.encode_range_image(batch['clean_lidar_range'])
    print(x)
    break

print("done")