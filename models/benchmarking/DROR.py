# TODO modify this to use open3d instead of PCL.
# TODO modify it to use nuscenes instead of the canadian adverse driving conditions dataset only. 

import os
# import pcl
import pickle
import logging
import argparse

import open3d as o3d

import numpy as np

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# # from ...utils.weather_augmentation.augment_pc import AugmentPointCloud
# from utils.weather_augmentation.augment_pc import AugmentPointCloud
# import utils

CADC_ROOT = Path().home() / 'datasets' / 'CADCD'
DENSE_ROOT = Path().home() / 'datasets' / 'DENSE' / 'SeeingThroughFog'


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='arg parser')
    # parser.add_argument('--dataset', type=str, default='DENSE', choices=['CADC', 'DENSE'],
    #                     help='specify the dataset to be processed')
    parser.add_argument('--alpha', type=float, default=0.08,
                        help='specify the horizontal angular resolution of the lidar')
    parser.add_argument('--crop', type=bool, default=True,
                        help='specify if the pointcloud should be cropped')
    parser.add_argument('--pkl', type=str, default='dense_infos_all.pkl',
                        help='specify the pkl file to be processed (only relevant for DENSE)')
    parser.add_argument('--sensor', type=str, default='hdl64', choices=['vlp32', 'hdl64'],
                        help='specify if the sensor type')
    parser.add_argument('--signal', type=str, default='strongest', choices=['strongest', 'last'],
                        help='specify if the signal type')

    args = parser.parse_args()

    if args.dataset == 'CADC':      # remove irrelevant arguments
        del args.pkl
        del args.sensor
        del args.signal
    else:                           # make sure correct sensor split is used
        if args.sensor == 'vlp32' and 'vlp32' not in args.pkl:
            args.pkl = args.pkl.replace('.pkl', '_vlp32.pkl')

    return args


def load_cadc_pointcloud(date: str, sequence: str, frame: str) -> np.ndarray:

    filename = CADC_ROOT / date / sequence / 'labeled' / 'lidar_points' / 'data' / frame

    pc = np.fromfile(filename, dtype=np.float32)
    pc = pc.reshape((-1, 4))

    pc[:, 3] = np.round(pc[:, 3] * 255)

    return pc


def load_dense_pointcloud(file: str, sensor: str, signal: str) -> np.ndarray:

    filename = DENSE_ROOT / f'lidar_{sensor}_{signal}' / f'{file}.bin'

    pc = np.fromfile(filename, dtype=np.float32)
    pc = pc.reshape((-1, 5))

    return pc


def get_cube_mask(pc: np.ndarray,
                  x_min: float = 3, x_max: float = 13,
                  y_min: float = -1, y_max: float = 1,
                  z_min: float = -1, z_max: float = 1) -> np.ndarray:

    x_mask = np.logical_and(x_min <= pc[:, 0], pc[:, 0] <= x_max)
    y_mask = np.logical_and(y_min <= pc[:, 1], pc[:, 1] <= y_max)
    z_mask = np.logical_and(z_min <= pc[:, 2], pc[:, 2] <= z_max)

    cube_mask = np.logical_and(x_mask, y_mask, z_mask)

    return cube_mask


def process_cadc(args: argparse.Namespace):

    log = args.logger

    dates = sorted(os.listdir(CADC_ROOT))

    stats_dict = {}
    lookup_dict = {}

    for date in dates:

        stats_dict[date] = {}
        lookup_dict[date] = {}

        sequences = sorted(os.listdir(CADC_ROOT / date))

        for sequence in sequences:

            # skip calib folder
            if sequence == 'calib':
                continue

            lookup_dict[date][sequence] = {}

            frames = sorted(os.listdir(CADC_ROOT / date / sequence / 'labeled' / 'lidar_points' / 'data'))
            pbar_frames = tqdm(range(len(frames)), desc=f'{date}/{sequence}')

            first_n_snow = -1
            min_n_snow = np.inf
            max_n_snow = 0
            avg_n_snow = 0

            min_n_cube = np.inf
            max_n_cube = 0
            avg_n_cube = 0

            for f in pbar_frames:

                frame = frames[f]

                pc = load_cadc_pointcloud(date, sequence, frame)

                cube_mask = get_cube_mask(pc)

                if args.crop:
                    pc = pc[cube_mask]

                keep_mask = dynamic_radius_outlier_filter(pc)

                snow_indices = (keep_mask == 0).nonzero()[0]#.astype(np.int16)
                lookup_dict[date][sequence][frame.replace('.bin', '')] = snow_indices

                # snow statistics

                n_snow = (keep_mask == 0).sum()

                if f == 0:
                    first_n_snow = n_snow

                if n_snow > max_n_snow:
                    max_n_snow = n_snow

                if n_snow < min_n_snow:
                    min_n_snow = n_snow

                avg_n_snow = (avg_n_snow * f + n_snow) / (f+1)

                # cube statistics

                n_cube = cube_mask.sum()

                if n_cube > max_n_cube:
                    max_n_cube = n_cube

                if n_cube < min_n_cube:
                    min_n_cube = n_cube

                avg_n_cube = (avg_n_cube * f + n_cube) / (f + 1)

                pbar_frames.set_postfix({'cube': f'{int(n_cube)}',
                                         'snow': f'{int(n_snow)}'})

            log.info(f'{date}/{sequence}   1st_snow: {int(first_n_snow):>4}, '
                                         f'min_snow: {int(min_n_snow):>4}, '
                                         f'avg_snow: {int(avg_n_snow):>4}, '
                                         f'max_snow: {int(max_n_snow):>4}, '
                                         f'min_cube: {int(min_n_cube):>4}, '
                                         f'avg_cube: {int(avg_n_cube):>4}, '
                                         f'max_cube: {int(max_n_cube):>4}')

            stats_dict[date][sequence] = {'1st': int(first_n_snow),
                                          'min': int(min_n_snow),
                                          'max': int(max_n_snow),
                                          'avg': int(avg_n_snow)}

    suffix = '_crop' if args.crop else ''

    stats_save_path = Path(__file__).parent.resolve() / 'data' / f'cadc_stats{suffix}.pkl'

    with open(stats_save_path, 'wb') as f:
        pickle.dump(stats_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    lookup_save_path = Path(__file__).parent.resolve() / 'data' / f'cadc_dror{suffix}.pkl'

    with open(lookup_save_path, 'wb') as f:
        pickle.dump(lookup_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def process_dense(args: argparse.Namespace):

    pcdet_path = Path(__file__).parent.resolve().parent.parent.parent.parent

    info_path = pcdet_path / 'data' / 'dense' / 'test_not_in_use' / args.pkl

    if not info_path.exists():
        info_path = str(info_path).replace('test_not_in_use', 'test_in_use')

    sensor = args.sensor
    signal = args.signal
    alpha = f'alpha_{args.alpha}'

    variant = 'crop' if args.crop else 'full'
    split = args.pkl.replace('.pkl', '').replace('dense_infos_', '').replace('_vlp32', '')

    dense_infos = []

    with open(info_path, 'rb') as i:
        infos = pickle.load(i)
        dense_infos.extend(infos)

    save_folder = Path.home() / 'Downloads' / 'DROR' / alpha / split / sensor / signal / variant
    save_folder.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(len(dense_infos)), desc='_'.join(str(save_folder).split('/')[-4:]))

    min_n_snow = np.inf
    max_n_snow = 0
    avg_n_snow = 0

    min_n_cube = np.inf
    max_n_cube = 0
    avg_n_cube = 0

    for i in pbar:

        info = dense_infos[i]

        file = dict(info)['point_cloud']['lidar_idx']
        save_path = save_folder / f'{file}.pkl'

        if save_path.exists():
            continue

        try:
            pc = load_dense_pointcloud(file=file, sensor=sensor, signal=signal)
        except FileNotFoundError:
            continue

        cube_mask = get_cube_mask(pc)

        if args.crop:
            pc = pc[cube_mask]

        if len(pc) == 0:
            snow_indices = []
            n_snow = 0
        else:
            keep_mask = dynamic_radius_outlier_filter(pc, alpha=args.alpha)
            snow_indices = (keep_mask == 0).nonzero()[0]#.astype(np.int16)
            n_snow = (keep_mask == 0).sum()

        with open(save_path, 'wb') as f:
            pickle.dump(snow_indices, f, protocol=pickle.HIGHEST_PROTOCOL)

        # snow statistics

        if n_snow > max_n_snow:
            max_n_snow = n_snow

        if n_snow < min_n_snow:
            min_n_snow = n_snow

        avg_n_snow = (avg_n_snow * i + n_snow) / (i + 1)

        # cube statistics

        n_cube = cube_mask.sum()

        if n_cube > max_n_cube:
            max_n_cube = n_cube

        if n_cube < min_n_cube:
            min_n_cube = n_cube

        avg_n_cube = (avg_n_cube * i + n_cube) / (i + 1)

        pbar.set_postfix({'cube': f'{int(n_cube)}',
                          'snow': f'{int(n_snow)}'})


# adapted from https://github.com/mpitropov/cadc_devkit/blob/master/other/filter_pointcloud.py#L13-L50
def dynamic_radius_outlier_filter(pc: np.ndarray, alpha: float = 0.16, beta: float = 3.0,
                                  k_min: int = 3, sr_min: float = 0.04) -> np.ndarray:
    """
    This function removes points from the point cloud that are too
    isolated from their neighbours. It uses a dynamic search radius for 
    each point, based on its distance from the sensor and some parameters.

    :param pc:      pointcloud
    :param alpha:   horizontal angular resolution of the lidar
    :param beta:    multiplication factor
    :param k_min:   minimum number of neighbors
    :param sr_min:  minumum search radius

    :return:        mask [False = snow, True = no snow]
    """

    # pc = pcl.PointCloud(pc[:, :3])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:,:3])

    # create KDTree
    # kd_tree = pc.make_kdtree_flann()
    '''
    KD-Trees are efficient data structures for nearest-neighbour searches
    allowing for faster queries compared to brute force methods. 
    '''
    kd_tree = o3d.geometry.KDTreeFlann(pcd)

    num_points = len(pc)
    # num_points = pc.size

    # initialize mask with False
    # this mask will keep track of which points to keep.
    mask = np.zeros(num_points, dtype=bool)
    k = k_min + 1

    for i in range(num_points):

        x = pc[i][0]
        y = pc[i][1]

        # compute the distance from the origin
        r = np.linalg.norm([x, y], axis=0)

        # compute the search radius based on the angular resolution and scaling factor.
        sr = alpha * beta * np.pi / 180 * r

        # ensure a minimum search radius.
        if sr < sr_min:
            sr = sr_min

        # find the indices of the k nearest points and their squared distances. 
        # [_, sqdist] = kd_tree.nearest_k_search_for_point(pc, i, k)
        [_, _, sqdist] = kd_tree.search_knn_vector_3d(pcd.points[i], k)

        neighbors = -1      # start at -1 since it will always be its own neighbour
        # count how many of them are within the search radius. 
        for val in sqdist:
            if np.sqrt(val) < sr:
                neighbors += 1

        # if it has more than the minimum, then we set the mask to true. 
        if neighbors >= k_min:
            mask[i] = True  # no snow -> keep

    return mask


if __name__ == '__main__':

    # load the pointcloud: 
    point_cloud_array = np.load("../augmented_pointcloud.npy")
    point_cloud_array = point_cloud_array[:,0:3]

    # they used a Velodyne HDL32E, 20Hz capture freq, 32 beams, 1080 points per ring, 32 channels.
    # assuming a 360 degree horizontal FOV with ~1800 points per rotation, not sure if this is right
    # then we have 360 deg / 1800 ~= 0.2 deg
    # try 360 / 1080 = 0.33
    alpha = 0.33
    # beta scales the search radius depending on distance from the sensor. 
    # a typical range is 2.0 to 5.0, larger means more variation in neighbourhood size at distance.
    beta = 5.0
    # minimum number of neighbours. 
    k_min = 3
    # the minimum search radius
    # it should be 1-2x your LIDAR point spacing near the origin. 
    sr_min = 0.001 # 0.04

    mask = dynamic_radius_outlier_filter(point_cloud_array, alpha=alpha, beta=beta, k_min=k_min, sr_min=sr_min)
    filtered_points = point_cloud_array[mask]

    # load the original point cloud
    original_point_cloud_array = np.load("../original_pointcloud.npy")
    original_point_cloud_array = original_point_cloud_array[:,0:3]

    # compute the chamfer distance
    from utils.weather_augmentation.analyze_pc import AnalyzePointCloud
    pc_metrics = AnalyzePointCloud()
    print("chamfer DROR: ", pc_metrics.chamfer_distance(filtered_points, original_point_cloud_array))
    print("chamfer noisy: ", pc_metrics.chamfer_distance(point_cloud_array, original_point_cloud_array))

    print("voxel diff DROR: ", pc_metrics.density_histogram_l1(filtered_points,original_point_cloud_array))
    print("voxel diff noisy: ", pc_metrics.density_histogram_l1(point_cloud_array,original_point_cloud_array))

    print("pointwise density diff DROR: ", pc_metrics.local_density_consistency(filtered_points,original_point_cloud_array))
    print("pointwise density diff noisy: ", pc_metrics.local_density_consistency(point_cloud_array,original_point_cloud_array))

    pcd_augmented = o3d.geometry.PointCloud()
    pcd_augmented.points = o3d.utility.Vector3dVector(point_cloud_array[:,:3])
    pcd_augmented.paint_uniform_color([1,0,0]) #red

    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(original_point_cloud_array)
    pcd_original.paint_uniform_color([0, 0, 1])   # blue

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points[:,:3])
    # display the point cloud: 
    filtered_pcd.paint_uniform_color([0, 1, 0])   # green
    
    
    # o3d.visualization.draw_geometries([pcd_augmented, pcd_original , filtered_pcd], window_name=f"DROR Outliers")
    o3d.visualization.draw_geometries([pcd_original ,pcd_augmented, filtered_pcd], window_name=f"DROR Outliers")
    # o3d.visualization.draw_geometries([pcd_augmented , filtered_pcd], window_name=f"DROR Outliers")


    ## --------- deprecated stuff that came in the file ----------

    # # parse the arguments. 
    # arguments = parse_args()

    # # datetime object containing current date and time
    # now = datetime.now()

    # # dd/mm/YY_H:M:S
    # dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    # this_file_location = Path(__file__).parent.resolve()

    # # setting up logging and console outputs.
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s %(levelname)8s  %(message)s', "%Y-%m-%d %H:%M:%S")

    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # console.setFormatter(formatter)

    # file_handler = logging.FileHandler(filename=this_file_location / 'logs' / f'{arguments.dataset}_{dt_string}.log')
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)

    # logger.addHandler(file_handler)
    # logger.addHandler(console)

    # arguments.logger = logger

    # # so it's the process dense and process cadc that will have the bulk of the processing code.

    # for c in [True, False]: # process with cropping and no cropping.

        # arguments.crop = c

        # # i believe this is specific to their dataset. 
        # if arguments.dataset == 'DENSE':

        #     for se in ['hdl64', 'vlp32']:

        #         arguments.sensor = se

        #         for si in ['strongest', 'last']:

        #             arguments.signal = si

        #             for key, value in arguments.__dict__.items():
        #                 if value is not None and key != 'logger':
        #                     logger.info(f'{key:<10}: {value}')

        #             process_dense(args=arguments)

        # else:

        #     for key, value in arguments.__dict__.items():
        #         if value is not None:
        #             logger.info(f'{key:<10}: {value}')

        #     process_cadc(args=arguments)
