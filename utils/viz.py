import matplotlib.pyplot as plt

def visualize_radar_lidar_bev(radar_pc, lidar_pc, save_path=None):
    """
    Visualizes the Bird's Eye View (BEV) of merged radar & LiDAR point clouds.

    Args:
        radar_pc (torch.Tensor): Radar point cloud (shape: [N, 19], where first 3 dims are (x, y, z)).
        lidar_pc (torch.Tensor): LiDAR point cloud (shape: [M, 3]).
        save_path (str, optional): If provided, saves the figure instead of displaying.
    """
    radar_points = radar_pc[:, :3].cpu().numpy()  # Extract (x, y, z)
    lidar_points = lidar_pc[:, :3].cpu().numpy()  # Extract (x, y, z)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot LiDAR points (black dots)
    ax.scatter(lidar_points[:, 0], lidar_points[:, 1], c='black', s=0.2, label='LiDAR')

    # Plot Radar points (red dots)
    ax.scatter(radar_points[:, 0], radar_points[:, 1], c='red', s=5, label='Radar')

    ax.set_xlim(-50, 50)  # Adjust based on LiDAR range
    ax.set_ylim(-50, 50)  
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('BEV of LiDAR & Merged Radar Data')
    ax.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved BEV visualization to {save_path}")
    else:
        plt.show()

def visualize_pc_bev(pc, type="torch", save_path=None):
    """
    Visualizes the Bird's Eye View (BEV) of merged radar & LiDAR point clouds.

    Args:
        pc (torch.Tensor): Point cloud (shape: [M, D]).
        save_path (str, optional): If provided, saves the figure instead of displaying.
    """

    points = pc[:, :3]  # Extract (x, y, z)

    if type == "torch":
        points = points.cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot LiDAR points (black dots)
    ax.scatter(points[:, 0], points[:, 1], c='black', s=0.2, label='Points')

    ax.set_xlim(-50, 50)  # Adjust based on LiDAR range
    ax.set_ylim(-50, 50)  
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('BEV of PC Data')
    ax.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved BEV visualization to {save_path}")
    else:
        plt.show()