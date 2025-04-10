
import sys
import os


if __name__ == '__main__':
    sys.path.append(os.path.dirname(__file__))

    from augment_pc import AugmentPointCloud
    from analyze_pc import AnalyzePointCloud


    apc = AugmentPointCloud()
    analyze = AnalyzePointCloud()

    pc_original = apc.load_pointcloud("/Users/williamdormer/datasets/nuScenes/sweeps/LIDAR_TOP/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801007396978.pcd.bin")
    pc_augmented = apc.augment("/Users/williamdormer/datasets/nuScenes/sweeps/LIDAR_TOP/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801007396978.pcd.bin")

    # analyze.visualize(pc_augmented)

    # analyze.visualize(pc_original, pc_augmented)

