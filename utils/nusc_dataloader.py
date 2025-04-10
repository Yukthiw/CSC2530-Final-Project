import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pyquaternion import Quaternion
from functools import reduce
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import create_splits_scenes

from utils.radar_encoder_utils import Vox_util
from utils.lidar_encoder_utils import Lidar_to_range_image

class NuscData(Dataset):
    def __init__(self, nusc, noisy_lidar_dataroot: str, is_train: bool, nsweeps: int = 1,
                 use_radar_filters: bool = True):
        """
        Dataset for loading NuScenes radar + LiDAR data.
        
        Args:
            nusc: NuScenes dataset instance.
            noisy_lidar_dataroot (str): Path to augmented lidar files.
            is_train (bool): Whether to load training or validation data.
            nsweeps (int): Number of past radar sweeps to aggregate.
            use_radar_filters (bool): Apply NuScenes' default radar filtering.
        """
        self.nusc = nusc
        self.is_train = is_train
        self.nsweeps = nsweeps
        self.use_radar_filters = use_radar_filters
        self.dataroot = self.nusc.dataroot
        self.augmented_pc_dataroot = noisy_lidar_dataroot

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()
        self.voxelizer = self.init_vox()
        self.range_img = Lidar_to_range_image()
    
    def init_vox(self):
        scene_centroid_x = 0.0
        scene_centroid_y = 1.0
        scene_centroid_z = 0.0

        scene_centroid_py = np.array([scene_centroid_x,
                                    scene_centroid_y,
                                    scene_centroid_z]).reshape([1, 3])
        scene_centroid = torch.from_numpy(scene_centroid_py).float()

        XMIN, XMAX = -50, 50
        ZMIN, ZMAX = -50, 50
        YMIN, YMAX = -5, 5
        bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
        Z, Y, X = 200, 8, 200
        return Vox_util(Z, Y, X, scene_centroid, bounds)

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
        radar_sensors = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
                        'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']

        all_radar_points = []

        # Reference time (same across all radar sensors)
        ref_sd_token = rec['data']['RADAR_FRONT']
        ref_sd_rec = self.nusc.get('sample_data', ref_sd_token)
        ref_time = 1e-6 * ref_sd_rec['timestamp']  # in seconds

        # Computing Ego frame -> Lidar frame transform
        ref_lidar_token = rec['data']['LIDAR_TOP']
        ref_lidar_rec = self.nusc.get('sample_data', ref_lidar_token)
        lidar_calib = self.nusc.get('calibrated_sensor', ref_lidar_rec['calibrated_sensor_token'])
        ego_to_lidar = transform_matrix(lidar_calib['translation'], Quaternion(lidar_calib['rotation']), inverse=True)

        for sensor in radar_sensors:
            if sensor not in rec['data']:
                continue

            radar_sample_data = self.nusc.get('sample_data', rec['data'][sensor])
            current_sd_rec = radar_sample_data

            for _ in range(nsweeps):
                if current_sd_rec is None:
                    break

                radar_pc = RadarPointCloud.from_file(os.path.join(self.dataroot, current_sd_rec['filename']))

                if self.use_radar_filters:
                    RadarPointCloud.default_filters()
                else:
                    RadarPointCloud.disable_filters()

                radar_pc.remove_close(min_distance)

                # Transform radar points to Ego frame
                radar_calib = self.nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
                radar_to_ego = transform_matrix(radar_calib['translation'], Quaternion(radar_calib['rotation']))
                radar_pc_hom = np.vstack((radar_pc.points[:3, :], np.ones((1, radar_pc.nbr_points()))))
                radar_pc_lidar_frame = (ego_to_lidar @ radar_to_ego @ radar_pc_hom)[:3, :] # Probably a smarter way of doing this but whatever

                radar_metadata = radar_pc.points[3:, :]

                # Time lag feature
                current_time = 1e-6 * current_sd_rec['timestamp']
                time_lag = ref_time - current_time
                times = time_lag * np.ones((1, radar_pc.nbr_points()))

                # Combine everything to (19, N)
                new_radar_points = np.vstack((radar_pc_lidar_frame, radar_metadata, times))
                all_radar_points.append(new_radar_points)

                if current_sd_rec['prev'] == '':
                    break
                else:
                    current_sd_rec = self.nusc.get('sample_data', current_sd_rec['prev'])

        if len(all_radar_points) > 0:
            radar_points_merged = np.hstack(all_radar_points)  # (19, N_total)
        else:
            radar_points_merged = np.zeros((19, 0))

        return torch.Tensor(radar_points_merged.T)  # (N_total, 19)

    def get_gt_lidar_data(self, rec):
        """
        Loads the LiDAR point cloud and returns the data IN LIDAR FRAME.
        Args:
            rec: NuScenes sample record.
        Returns:
            torch.Tensor: Transformed LiDAR point cloud (shape: `[M, 5]`).
        """
        # Load LiDAR point cloud 
        lidar_sample_data = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        lidar_path = os.path.join(self.dataroot, lidar_sample_data['filename'])
        lidar_pc = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5) # Shape [M, 5]

        return torch.Tensor(lidar_pc)

    def get_perturbed_lidar_data(self, rec):
        """
        Loads augmented LiDAR point cloud matching ground truth.
        """
        lidar_sample_data = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        original_path = os.path.join(self.dataroot, lidar_sample_data['filename'])
        base_filename = os.path.splitext(os.path.splitext(os.path.basename(original_path))[0])[0] # Lidar pc filenames are <file_name>.pcd.bin
        aug_path = os.path.join(self.augmented_pc_dataroot, base_filename + '.npy')
        if os.path.exists(aug_path):
            lidar_pc = np.load(aug_path)  # Shape [M, 5]
            return torch.Tensor(lidar_pc)
        else:
            raise FileExistsError(f"{aug_path} does not exist.")
            
    def trim_pad_pc(self, pc: torch.Tensor, target_size: int):
        # Trimming/Padding radar points
        num_points = pc.shape[0]

        if num_points > target_size:
            # Trim excess points (randomly)
            idx = torch.randperm(num_points)[:target_size]  # Random subset
            radar_data = pc[idx]

        elif num_points < target_size:
            # Pad with zeros if fewer points exist
            pad_needed = target_size - num_points
            pad_tensor = torch.zeros((pad_needed, pc.shape[1]), dtype=pc.dtype)
            radar_data = torch.cat([pc, pad_tensor], dim=0)
        else: 
            radar_data = pc
        
        return radar_data
            
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
            gt_lidar_pc = self.get_gt_lidar_data(rec)
            noisy_lidar_pc = self.get_perturbed_lidar_data(rec)

            # Trimming/Padding radar points + Voxelizing (Need to be fixed dimension for voxel util)
            RADAR_SIZE = 700 * self.nsweeps
            radar_data = self.trim_pad_pc(radar_pc, RADAR_SIZE)
            radar_vox = self.voxelizer.voxelize(radar_data.unsqueeze(0))

            gt_lidar_range = self.range_img(gt_lidar_pc)
            noisy_lidar_range = self.range_img(noisy_lidar_pc)

            # Trimming/Padding lidar points
            LIDAR_SIZE = 35000
            gt_lidar_data = self.trim_pad_pc(gt_lidar_pc, LIDAR_SIZE)
            noisy_lidar_data = self.trim_pad_pc(noisy_lidar_pc, LIDAR_SIZE)
            return {
                'radar_pc': radar_data,  # Shape: (700, 19)
                'radar_vox': radar_vox, # Tuple (3, )
                'clean_lidar_pc': gt_lidar_data,   # Shape: (35000, 3)
                'clean_lidar_range': gt_lidar_range, # Shape: (2, 1024, 32)
                'noisy_lidar_pc': noisy_lidar_data,   # Shape: (35000, 3)
                'noisy_lidar_range': noisy_lidar_range, # Shape: (2, 1024, 32)
            }

    def __len__(self):
        return len(self.ixes)

    def __str__(self):
        return f"NuscData: {len(self)} samples. Split: {'train' if self.is_train else 'val'}."

