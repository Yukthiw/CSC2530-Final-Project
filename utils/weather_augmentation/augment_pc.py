import numpy as np

# import sys
# import os

# sys.path.append(os.path.dirname(__file__))

from tools.snowfall.simulation import augment
from tools.snowfall.sampling import snowfall_rate_to_rainfall_rate, compute_occupancy

dtype = np.float32
num_features = 5

'''
Define a class that handles the weather purturbation logic.
'''
class AugmentPointCloud:
    def __init__(self):
        # define the parameters relevant to the augment function.
        self.dataset = None
        self.min_value = None
        self.max_value = None
        self.num_features = None
        self.extension = None
        self.intensity_multiplier = None

        # other parameters that were just hardcoded
        self.noise_floor = 0.7
        self.beam_divergence = 0.003 # (RAD)
        self.mode = "gunn" 
        self.snowfall_rate = '0.5'
        self.terminal_velocity = '0.2'

        self.MIN_DIST = 3 # m, to hide "the ring" around the sensor.

        self.min_height = -400 # cm
        self.max_distance = 80 # m

        self.dtype = np.float32

        # load the nuscenes parameters
        self.set_nuscenes()

    def set_nuscenes(self):
        '''
        sets the parameters to follow the nuScenes dataset.
        '''
        self.dataset = 'nuScenes'
        self.min_value = 0
        self.max_value = 31
        self.num_features = 5
        self.extension = 'bin'
        self.intensity_multiplier = 1
    
    def load_pointcloud(self, filename:str) -> np.ndarray:
        '''
        load a pointcloud from a file, and return the pointcloud.

        input:
            filename: the path to the pointcloud file.
        output:
            an ndarray containing the pointcloud information. 
        '''

        pc = np.fromfile(filename, dtype=self.dtype)
        pc = pc.reshape((-1, num_features))
        pc[:, 3] = np.round(pc[:, 3] * self.intensity_multiplier)
        return pc

    def augment(self, filename:str , savepath:str = None):
        '''
        Augments and saves a given pointcloud

        inputs: 
            filename: the file path of the pointcloud to augment
            savepath: the path to save the file, if None, then it returns the point cloud instead of saving it. 
        '''
        # load the pointcloud
        pc = self.load_pointcloud(filename)

        # mask the points to the relevant region

        # remove points in ring around sensor.
        min_dist_mask = np.linalg.norm(pc[:, 0:3], axis=1) > self.MIN_DIST
        pc = pc[min_dist_mask, :]

        # remove points beyond the max distance
        max_dist_mask = np.linalg.norm(pc[:, 0:3], axis=1) < self.max_distance
        pc = pc[max_dist_mask, :]

        # get ride of points below the max height. 
        min_height_mask = pc[:, 2] > (self.min_height / 100)  # in m
        pc = pc[min_height_mask, :]
        
        rain_rate = snowfall_rate_to_rainfall_rate(float(self.snowfall_rate), float(self.terminal_velocity))
        already_augmented = False
        occupancy = compute_occupancy(float(self.snowfall_rate), float(self.terminal_velocity))

        # TODO make this more modular
        # snowflake_file_prefix = f'snowflake_patterns/npy/{self.mode}_{rain_rate}_{occupancy}'
        snowflake_file_prefix = f'{self.mode}_{rain_rate}_{occupancy}'

        # np.save('original_pointcloud.npy', pc)

        stats, pc = augment(pc=pc, only_camera_fov=False,
                                        particle_file_prefix=snowflake_file_prefix, noise_floor=self.noise_floor,
                                        beam_divergence=float(np.degrees(self.beam_divergence)),
                                        shuffle=True, show_progressbar=True)

        num_attenuated, num_removed, avg_intensity_diff = stats

        num_unchanged = (pc[:, 4] == 0).sum()
        num_scattered = (pc[:, 4] == 2).sum()

        print(f"num_unchanged: {num_unchanged}")
        print(f"num_scattered: {num_scattered}")
        print(f"num_attenuated: {num_attenuated}")
        print(f"num_removed: {num_removed}")
        print(f"avg_intensity_diff: {avg_intensity_diff}")

        if savepath == None:
            return pc
        else:
            np.save(savepath, pc)
            # np.save('augmented_pointcloud.npy', pc)




    
