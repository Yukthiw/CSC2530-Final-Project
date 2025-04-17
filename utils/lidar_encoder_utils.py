from typing import Any
import numpy as np
import torch 
import os 
from pathlib import Path

    
class Lidar_to_range_image():
    def __init__(self):
        self.file_paths = None
        self.range_fill_value = np.array([100, 0])
        self.height = np.array([
            -0.00216031, -0.00098729, -0.00020528,  0.00174976,  0.0044868, -0.00294233,
            -0.00059629, -0.00020528,  0.00174976, -0.00294233, -0.0013783,  0.00018573,
             0.00253177, -0.00098729,  0.00018573,  0.00096774, -0.00411535, -0.0013783,
             0.00018573,  0.00018573, -0.00294233, -0.0013783, -0.00098729, -0.00020528,
             0.00018573,  0.00018573,  0.00018573, -0.00020528,  0.00018573,  0.00018573,
             0.00018573,  0.00018573
        ], dtype=np.float32)
        self.zenith = np.array([
             0.18670577,  0.16324536,  0.13978495,  0.11632454,  0.09286413,  0.07018573,
             0.04672532,  0.02326491, -0.0001955,  -0.0228739,  -0.04633431, -0.06979472,
            -0.09325513, -0.11593353, -0.13939394, -0.16285435, -0.18553275, -0.20899316,
            -0.23245357, -0.25591398, -0.27859238, -0.30205279, -0.3255132,  -0.34897361,
            -0.37243402, -0.39589443, -0.41935484, -0.44203324, -0.46549365, -0.48895406,
            -0.51241447, -0.53587488
        ], dtype=np.float32)
        self.width = 1024
        self.incl = -self.zenith
        self.H = 32
        self.mean = 50.
        self.std = 50.
        self.range_limit = 90 #190.0


    def get_pts(self, pts_path):
        pts = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 5)
        pts[:, 3] = pts[:, 3] / 255.0
        return pts
    
    # def get_pth_path(self, pts_path):
    #     return pts_path.replace('sweeps', 'sweeps_range').replace('.bin', '.pth')

    def generate_range_image(self, pc):
        depth = np.linalg.norm(pc[:,:3], 2, axis=1)
        mask = depth > 2.0
        pc = pc[mask, :]
        row_inds = self.get_row_inds(pc)

        azi = np.arctan2(pc[:,1], pc[:,0])
        col_inds = self.width - 1.0 + 0.5 - (azi + np.pi) / (2.0 * np.pi) * self.width
        col_inds = np.round(col_inds).astype(np.int32)
        col_inds[col_inds == self.width] = self.width - 1
        col_inds[col_inds < 0] = 0
        empty_range_image = np.full((self.H, self.width, 2), -1, dtype = np.float32)
        pc[:,2] -= self.height[row_inds]
        point_range = np.linalg.norm(pc[:,:3], axis = 1, ord = 2)
        point_range[point_range > self.range_fill_value[0]] = self.range_fill_value[0]

        order = np.argsort(-point_range)
        point_range = point_range[order] 
        pc = pc[order]
        row_inds = row_inds[order]
        col_inds = col_inds[order]

        empty_range_image[row_inds, col_inds, :] = np.concatenate([point_range[:,None], pc[:,3:4]], axis = 1)

        return empty_range_image
    
    def get_row_inds(self, pc):
        row_inds = 31 - pc[:, 4].astype(np.int32) # nuscenes already has the row_inds
        return row_inds
    
        
    @staticmethod
    def fill_noise(data, miss_inds, width):
        data_shift1pxl = data[:, list(range(1, width)) + [0, ], :]
        data[miss_inds, :] = data_shift1pxl[miss_inds, :]
        return data
    
    def process_miss_value(self, range_image):
        _, width, _ = range_image.shape
        miss_inds = range_image[:, :, 0] == -1

        range_image = self.fill_noise(range_image, miss_inds, width)

        still_miss_inds = range_image[:, :, 0] == -1

        range_image[still_miss_inds, :] = self.range_fill_value 

        # How much are the intensity and elongation of car windows
        # range_image[car_window_mask, :] = np.array([0, 0])

        return range_image

    def normalize(self, range_image):
        range_image[..., 0] = (range_image[..., 0] - self.mean) / self.std
        return range_image
    
    
    def to_pc_torch(self, range_images):
        '''
        range_images: Bx2xWxH
        output:
            point_cloud: BxNx4
        '''
        device = range_images.device
        incl_t = torch.from_numpy(self.incl).to(device)
        height_t = torch.from_numpy(self.height).to(device)
        batch_size, channels, width_dim, height_dim = range_images.shape

        # Extract point range and remission
        point_range = range_images[:, 0, :, :] * self.std + self.mean # BxWxH
        if range_images.shape[1] > 1:
            remission = range_images[:, 1, :, :].reshape(batch_size, -1)

        r_true = point_range 

        r_true[r_true<0] = self.range_fill_value[0]

        # Calculate z
        z = (height_t[None,None,:] - r_true * torch.sin(incl_t[None,None,:])).reshape(batch_size, -1)

        # Calculate xy_norm
        xy_norm = r_true * torch.cos(incl_t[None,None,:])

        # Calculate azi
        width = width_dim
        azi = (width - 0.5 - torch.arange(0, width, device=device)) / width * 2. * torch.pi - torch.pi

        # Calculate x and y
        x = (xy_norm * torch.cos(azi[None,:,None])).reshape(batch_size, -1)
        y = (xy_norm * torch.sin(azi[None,:,None])).reshape(batch_size, -1)

        # Concatenate the arrays to create the point cloud
        if range_images.shape[1] > 1:
            point_cloud = torch.stack([x, y, z, remission], dim=2)
        else:
            point_cloud = torch.stack([x, y, z], dim=2)
        return point_cloud

    def remove_outliers(self, point_cloud):
        '''
        in: point cloud Nx4
        out: point cloud Nx4
        '''
        depth = torch.norm(point_cloud[:, :3], dim=1)
        mask = depth < self.range_limit
        return point_cloud[mask]

    def __call__(self, lidar_pc):
            
            
            #pth_path = self.get_pth_path(pts_path)
            #if os.path.exists(pth_path):
            #    range_image = torch.load(pth_path)
            #else: 
            lidar_pc = lidar_pc.cpu().numpy().astype(np.float32)
            lidar_pc[:, 3] = lidar_pc[:, 3] / 255.0
            range_image = self.generate_range_image(lidar_pc)
            range_image  = self.process_miss_value(range_image)
            range_image = self.normalize(range_image)
            range_image = torch.from_numpy(range_image).permute(2, 1, 0) # (C, W, H)
            range_image = range_image[:2]
            #pth_path = Path(pth_path)
            #os.makedirs(str(pth_path.parent), exist_ok=True)
            #torch.save(range_image, pth_path)
            return range_image