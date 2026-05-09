import numpy as np
import open3d as o3d
from .base import Detection, Detector


class EuclideanDetector(Detector):
    def __init__(
        self,
        eps: float = 0.5,
        min_points: int = 10,
        min_h: float = 0.3, max_h: float = 3.5,
        min_w: float = 0.3, max_w: float = 3.5,
        min_l: float = 0.3, max_l: float = 6.0,
        max_center_z: float = 1.0,
    ):
        self.eps = eps
        self.min_points = min_points
        self.min_h, self.max_h = min_h, max_h
        self.min_w, self.max_w = min_w, max_w
        self.min_l, self.max_l = min_l, max_l
        self.max_center_z = max_center_z

    def detect(self, lidar_points: np.ndarray) -> list[Detection]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3])

        labels = np.array(pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points))

        detections = []
        for label in np.unique(labels):
            if label == -1:
                continue

            cluster_points = lidar_points[labels == label]
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points[:, :3])

            try:
                obb = cluster_pcd.get_oriented_bounding_box()
            except Exception:
                continue

            center = np.asarray(obb.center)
            extent = np.asarray(obb.extent)
            R = np.asarray(obb.R)

            # Identify the most vertical axis (most aligned with Z)
            vertical_idx = int(np.argmax(np.abs(R[2, :])))
            horiz_idxs = [i for i in range(3) if i != vertical_idx]

            h = extent[vertical_idx]
            horiz_extents = sorted([extent[i] for i in horiz_idxs], reverse=True)
            l, w = horiz_extents[0], horiz_extents[1]

            if center[2] > self.max_center_z:
                continue
            if not (self.min_w <= w <= self.max_w and self.min_l <= l <= self.max_l and self.min_h <= h <= self.max_h):
                continue

            # Yaw from the longer horizontal principal axis
            long_axis_idx = horiz_idxs[0] if extent[horiz_idxs[0]] >= extent[horiz_idxs[1]] else horiz_idxs[1]
            rotation_y = float(np.arctan2(R[1, long_axis_idx], R[0, long_axis_idx]))

            detections.append(Detection(
                height=h, width=w, length=l,
                x=float(center[0]),
                y=float(center[1]),
                z=float(center[2] - h / 2.0),  # z at bottom of box
                rotation_y=rotation_y,
            ))

        return detections