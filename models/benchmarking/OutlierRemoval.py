import numpy as np
import open3d as o3d

'''
See the following cods for more information about open3d's outlier removal methods.
'''
# https://www.open3d.org/html/tutorial/Advanced/pointcloud_outlier_removal.html

# class OutlierRemoval:
#     def __init__(self):

#         self.icp_threshold = 0.02 # Maxium correspondence point distance (tune as needed)
#         self.icp_trans_init = np.eye(4) # the initial transformation (identity)
#         pass
    
# load the augmented pointcloud
# pcd = o3d.io.read_point_cloud("../augmented_pointcloud.npy")
point_cloud_array = np.load("../augmented_pointcloud.npy")
point_cloud_array = point_cloud_array[:,0:3]
if point_cloud_array.shape[1] != 3:
    print("point_cloud_array.shape: ", point_cloud_array.shape)
    raise ValueError("Input point cloud array must have shape (N,3)")

pcd_augmented = o3d.geometry.PointCloud()
pcd_augmented.points = o3d.utility.Vector3dVector(point_cloud_array)

# load the original point cloud
original_point_cloud_array = np.load("../original_pointcloud.npy")
original_point_cloud_array = original_point_cloud_array[:,0:3]
pcd_original = o3d.geometry.PointCloud()
pcd_original.points = o3d.utility.Vector3dVector(original_point_cloud_array)
pcd_original.paint_uniform_color([0, 0, 1])   # blue


# apply a radius based outlier removal
# Here, nb_points is the minimum number of points within the given radius to be considered a neighbor
# radius is the sphere within which we look for neighboring points
# filtered_pcd, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
nb_points = 1
radius = 1
radius_filtered_pcd, ind = pcd_augmented.remove_radius_outlier(nb_points=1, radius=1)
radius_filtered_pcd.paint_uniform_color([0, 1, 0])   # green

pcd_augmented.paint_uniform_color([1,0,0]) # red
# Visualize the result
# o3d.visualization.draw_geometries([pcd, radius_filtered_pcd], window_name=f"Radius Outliers, nb={nb_points}, radius={radius}")

# radius based struggles to detect correct far away points.


# print("Statistical oulier removal")
'''
nb_neighbors allows to specify how many neighbors are taken into account in order to calculate the average distance for a given point.
std_ratio allows to set the threshold level based on the standard deviation of the average distances across the point cloud. The lower this number the more aggressive the filter will be.
'''
nb_neighbours = 10
std_ratio = 0.1 # 2.0
stat_filtered_pcd, ind = pcd_augmented.remove_statistical_outlier(nb_neighbors=nb_neighbours, std_ratio=std_ratio)
stat_filtered_pcd.paint_uniform_color([0, 1, 0])   # green
o3d.visualization.draw_geometries([pcd_augmented, pcd_original, radius_filtered_pcd], window_name=f"Stat Outliers, nb={nb_neighbours}, std_ratio={std_ratio}")

# statistics based fails to remove the points close to the sensor. 

