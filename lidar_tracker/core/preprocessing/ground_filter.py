import numpy as np
import open3d as o3d

def remove_ground(points: np.ndarray, ground_threshold: float = 0.2) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    _, inlier_indices = pcd.segment_plane(distance_threshold=ground_threshold, ransac_n=3, num_iterations=1000)
    mask = np.zeros(points.shape[0], dtype=bool)
    mask[inlier_indices] = True
    return points[~mask]