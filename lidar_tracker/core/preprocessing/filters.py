import numpy as np
import open3d as o3d

def range_crop(points: np.ndarray, min_distance: float, max_distance: float) -> np.ndarray:
    """Crop points outside a specified distance range from the origin."""
    distances = np.linalg.norm(points[:, :3], axis=1)
    return points[(distances > min_distance) & (distances < max_distance)]

def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample points using a voxel grid filter."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    out_points = np.asarray(downsampled_pcd.points)
    out_points = np.hstack([out_points, np.ones((out_points.shape[0], 1))])  # Add intensity channel
    return out_points