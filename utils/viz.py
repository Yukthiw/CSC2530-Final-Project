import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


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

def visualize_pc_bev(pc, save_path=None):
    """
    Visualizes the Bird's Eye View (BEV) of merged radar & LiDAR point clouds.

    Args:
        pc (torch.Tensor): Point cloud (shape: [M, D]).
        save_path (str, optional): If provided, saves the figure instead of displaying.
    """
    points = pc[:, :3].cpu().numpy()  # Extract (x, y, z)
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
        
        
def plot_point_cloud_comparison(pc1, pc2, save_path="comparison_output.html", title1="PC 1", title2= "PC 2"):
    """
    Visualize and compare two point clouds using Plotly, with the fourth column used for color mapping.
    
    :param pc1: First point cloud (N1 x 3 or N1 x 4 numpy array)
    :param pc2: Second point cloud (N2 x 3 or N2 x 4 numpy array)
    :param save_path: Path to save the interactive Plotly visualization (HTML file)
    """
    # Ensure both point clouds are NumPy arrays
    pc1 = np.array(pc1)
    pc2 = np.array(pc2)

    # Convert point clouds to lists of (x, y, z) coordinates
    pc1_points = pc1[:, :3] if pc1.shape[1] >= 3 else pc1
    pc2_points = pc2[:, :3] if pc2.shape[1] >= 3 else pc2
    # Normalize the intensity values from the 4th column (if it exists)
    if pc1.shape[1] == 4:
        intensity1 = np.clip(pc1[:, 3] / np.max(pc1[:, 3]), 0, 1)
    else:
        intensity1 = np.zeros(pc1.shape[0])  # Default to black if no intensity data

    if pc2.shape[1] == 4:
        intensity2 = np.clip(pc2[:, 3] / np.max(pc2[:, 3]), 0, 1)
    else:
        intensity2 = np.zeros(pc2.shape[0])  # Default to black if no intensity data

    # Create traces for the point clouds, using intensity as color
    trace1 = go.Scatter3d(
        x=pc1_points[:, 0], 
        y=pc1_points[:, 1], 
        z=pc1_points[:, 2], 
        mode='markers', 
        marker=dict(size=2, color=intensity1, colorscale='Viridis', opacity=0.8),
        name=title1,
        showlegend=True
    )

    trace2 = go.Scatter3d(
        x=pc2_points[:, 0], 
        y=pc2_points[:, 1], 
        z=pc2_points[:, 2], 
        mode='markers', 
        marker=dict(size=2, color=intensity2, colorscale='CiVidis', opacity=0.8),
        name=title2,
        showlegend=True
    )

    # Create the layout for the Plotly figure
    layout = go.Layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        margin=dict(l=0, r=0, b=0, t=0),  # No margins around the plot
        title="Point Cloud Comparison",
        showlegend=True,
             # Adjust the layout to ensure it's visible above other elements
        legend=dict(
                x=-10.05,  # Position the legend outside the plot to the right
                y=50,     # Align it at the top of the plot
                traceorder='normal',
                orientation='v',
                font=dict(size=12),
                borderwidth=2,
                bordercolor='Black',
                bgcolor='rgba(255, 255, 255, 0.8)',  # Light background to make the legend readable
                itemclick='toggleothers',  # Allow clicking to toggle visibility
            ),
    )

    # Create the figure with both point clouds
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Save the interactive plot as an HTML file
    fig.write_html(save_path)
    print(f"Saved interactive point cloud comparison to {save_path}")