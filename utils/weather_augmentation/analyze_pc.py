import numpy as np
import open3d as o3d

class AnalyzePointCloud:
    def __init__(self):

        self.icp_threshold = 0.02 # Maxium correspondence point distance (tune as needed)
        self.icp_trans_init = np.eye(4) # the initial transformation (identity)
        pass
    def visualize(self, pc_original, pc_augmented=None):
        '''
        This renders the original and augmented pointclouds using open3D 

        inputs:
            pc_original: the numpy array of the original point cloud
            pc_augmented: the numpy array of the augmented point cloud, if none, just render the original point cloud
        '''

        # convert them to open3d pointcloud objects
        pcd_original = o3d.geometry.PointCloud()
        
        # convert the points to 3d vectors.
        pcd_original.points = o3d.utility.Vector3dVector(pc_original[:,0:3])

        # give each a unique color.
        pcd_original.paint_uniform_color([0, 1, 0])   # green

        if pc_augmented != None:
            pcd_augmented = o3d.geometry.PointCloud()
            pcd_augmented.points = o3d.utility.Vector3dVector(pc_augmented[:,0:3])
            pcd_augmented.paint_uniform_color([1, 0, 0])  # red

            o3d.visualization.draw_geometries([pcd_augmented, pcd_original],
                                   window_name="Pointcloud visualization",
                                   point_show_normal=False)
        else:
            o3d.visualization.draw_geometries([pcd_original],
                                   window_name="Pointcloud visualization",
                                   point_show_normal=False)

        # render the visualization.
        # o3d.visualization.draw_geometries([pcd_original, pcd_augmented],
        #                            window_name="Pointcloud visualization",
        #                            point_show_normal=False)
        # o3d.visualization.draw_geometries([pcd_augmented, pcd_original],
        #                            window_name="Pointcloud visualization",
        #                            point_show_normal=False)

    def icp(self, pc_original, pc_augmented):
        '''
        Computes the ICP (iterative closest point) matching between the two pointclouds, and returns error metrics

        return: 
            tuple of 
                fitness, closer to 1 is better
                inlier_rmse, lower is better
        '''

        # convert them to open3d pointcloud objects
        pcd_original = o3d.geometry.PointCloud()
        pcd_augmented = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(pc_original[:,0:3])
        pcd_augmented.points = o3d.utility.Vector3dVector(pc_augmented[:,0:3])
        
        # begin the ICP algorithm
        pcd1 = pcd_original
        pcd2 = pcd_augmented

        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd1, pcd2, self.icp_threshold, self.icp_trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )

        # Get fitness score (1.0 is perfect alignment)
        # print("ICP Fitness Score:", reg_p2p.fitness) # closer to 1 means better alignment
        # print("ICP RMSE:", reg_p2p.inlier_rmse) # lower is better

        return reg_p2p.fitness, reg_p2p.inlier_rmse

    def _avg_nearest_neighbor_dist(self,source, target_tree):
        '''
        finds the distance to the nearest neighbour in the other point cloud using kd_trees.
        '''
        distances = []
        for point in source.points:
            _, idx, dists = target_tree.search_knn_vector_3d(point, 1)
            distances.append(np.sqrt(dists[0]))
        return np.mean(distances)

    def chamfer_distance(self, pc_original, pc_augmented):
        '''
        Computes the chamfer distance between the two pointclouds

        measures the similarity between two point clouds by averaging the distances between each point in one cloud to its nearest neighbor in the other cloud, and vice versa
        '''

        # convert them to open3d pointcloud objects
        pcd_original = o3d.geometry.PointCloud()
        pcd_augmented = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(pc_original[:,0:3])
        pcd_augmented.points = o3d.utility.Vector3dVector(pc_augmented[:,0:3])

        pcd1_tree = o3d.geometry.KDTreeFlann(pcd_original)
        pcd2_tree = o3d.geometry.KDTreeFlann(pcd_augmented)

        cd_1 = self._avg_nearest_neighbor_dist(pcd_original, pcd2_tree)
        cd_2 = self._avg_nearest_neighbor_dist(pcd_augmented, pcd1_tree)
        
        return (cd_1 + cd_2) / 2





