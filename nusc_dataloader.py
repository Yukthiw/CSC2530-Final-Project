import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pyquaternion import Quaternion
from functools import reduce
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import create_splits_scenes

class NuscData(Dataset):
    def __init__(self, nusc, is_train: bool, nsweeps: int = 1,
                 use_radar_filters: bool = True, custom_dataroot: str = None):
        """
        Dataset for loading NuScenes radar + LiDAR data.
        
        Args:
            nusc: NuScenes dataset instance.
            is_train (bool): Whether to load training or validation data.
            data_aug_conf (dict): Data augmentation configuration.
            nsweeps (int): Number of past radar sweeps to aggregate.
            use_radar_filters (bool): Apply NuScenes' default radar filtering.
            custom_dataroot (str): Optional custom dataset root.
        """
        self.nusc = nusc
        self.is_train = is_train
        self.nsweeps = nsweeps
        self.use_radar_filters = use_radar_filters
        self.dataroot = self.nusc.dataroot if custom_dataroot is None else custom_dataroot

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

    def get_scenes(self):
        """
        Retrieves scene tokens based on dataset split (train/val).
        """
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
            'v1.0-test': {True: 'test', False: 'test'},
        }[self.nusc.version][self.is_train]
        
        return create_splits_scenes()[split]

    def prepro(self):
        """
        Filters dataset samples based on the selected split.
        """
        return [samp for samp in self.nusc.sample
                if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

    def get_radar_data(self, rec, nsweeps=1, min_distance=2.2):
        """
        Loads radar data from all 5 sensors, applies NuScenes filtering, merges sweeps,
        and transforms into the LiDAR coordinate system.

        Args:
            rec: NuScenes sample record.
            nsweeps (int): Number of past radar sweeps to aggregate.
            min_distance (float): Minimum allowed distance for radar points.

        Returns:
            torch.Tensor: Merged radar point cloud (shape: `[N, 19]`), aligned to the LiDAR frame.
        """
        radar_sensors = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
                         'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        
        all_radar_points = []

        for sensor in radar_sensors:
            if sensor not in rec['data']:
                continue

            # Load radar sample data
            radar_sample_data = self.nusc.get('sample_data', rec['data'][sensor])
            current_sd_rec = radar_sample_data  # Start with the latest sweep

            for _ in range(nsweeps):
                if current_sd_rec is None:
                    break  # No more sweeps available

                # Load radar point cloud
                radar_pc = RadarPointCloud.from_file(os.path.join(self.dataroot, current_sd_rec['filename']))

                # Apply NuScenes' default filtering
                if self.use_radar_filters:
                    RadarPointCloud.default_filters()
                else:
                    RadarPointCloud.disable_filters()

                # Remove close-range artifacts
                radar_pc.remove_close(min_distance)

                # Get radar sensor transformation (Radar → Ego)
                radar_calib = self.nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
                radar_to_ego = transform_matrix(radar_calib['translation'], Quaternion(radar_calib['rotation']))

                # Transform radar points to the Ego frame
                radar_pc_hom = np.vstack((radar_pc.points[:3, :], np.ones((1, radar_pc.nbr_points()))))
                radar_pc_ego = (radar_to_ego @ radar_pc_hom)[:3, :]

                # Preserve additional radar metadata (velocity, RCS, etc.)
                radar_metadata = radar_pc.points[3:, :]

                # Append transformed radar data with metadata
                new_radar_points = np.vstack((radar_pc_ego, radar_metadata))
                all_radar_points.append(new_radar_points)

                # Move to previous sweep (if available)
                if current_sd_rec['prev'] == '':
                    break
                else:
                    current_sd_rec = self.nusc.get('sample_data', current_sd_rec['prev'])

        # Merge all radar points into a single array
        if len(all_radar_points) > 0:
            radar_points_merged = np.hstack(all_radar_points)  # Shape: (19, N_total)
        else:
            radar_points_merged = np.zeros((19, 0))  # Empty placeholder

        return torch.Tensor(radar_points_merged.T)  # Convert to shape (N, 19)


    def get_lidar_data(self, rec):
        """
        Loads the LiDAR point cloud and transforms it to the ego-vehicle frame.
        Args:
            rec: NuScenes sample record.
        Returns:
            torch.Tensor: Transformed LiDAR point cloud (shape: `[M, 3]`).
        """
        # Load LiDAR point cloud 
        lidar_sample_data = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        lidar_path = os.path.join(self.dataroot, lidar_sample_data['filename'])
        lidar_pc = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]  # Only take XYZ

        # Get LiDAR → Ego transformation matrix
        lidar_calib = self.nusc.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])
        lidar_to_ego = transform_matrix(lidar_calib['translation'], Quaternion(lidar_calib['rotation']))

        # Convert to homogeneous coordinates (4D)
        lidar_pc_hom = np.hstack((lidar_pc, np.ones((lidar_pc.shape[0], 1))))  # Shape: [M, 4]

        # Apply LiDAR → Ego transformation
        lidar_pc_ego = (lidar_to_ego @ lidar_pc_hom.T)[:3, :].T  # Shape: [M, 3]

        return torch.Tensor(lidar_pc_ego)

    def __getitem__(self, index):
            """
            Retrieves a sample, loading radar and LiDAR data.

            Args:
                index (int): Sample index.

            Returns:
                dict: Dictionary with 'radar' and 'lidar' point clouds.
            """
            rec = self.ixes[index]
            radar_pc = self.get_radar_data(rec, nsweeps=self.nsweeps, min_distance=2.2)
            lidar_pc = self.get_lidar_data(rec)

            # Trimming/Padding radar points
            V = 700 * self.nsweeps
            num_points = radar_pc.shape[0]

            if num_points > V:
                # Trim excess points (randomly)
                idx = torch.randperm(num_points)[:V]  # Random subset
                radar_data = radar_pc[idx]

            elif num_points < V:
                # Pad with zeros if fewer points exist
                pad_needed = V - num_points
                pad_tensor = torch.zeros((pad_needed, radar_data.shape[1]), dtype=lidar_pc.dtype)
                radar_data = torch.cat([radar_data, pad_tensor], dim=0)
            else: 
                radar_data = radar_pc
            return {
                'radar': radar_data,  # Shape: (700*nsweeps, 19)
                'lidar': lidar_pc   # Shape: (M, 3)
            }

    def __len__(self):
        return len(self.ixes)

    def __str__(self):
        return f"NuscData: {len(self)} samples. Split: {'train' if self.is_train else 'val'}."

