import numpy as np
import open3d as o3d
from .base import Detection, Detector

class EuclideanDetector(Detector):
    def __init__(self, eps: float = 0.5, min_points: int = 10):
        self.eps = eps
        self.min_points = min_points

    def detect(self, lidar_points: np.ndarray) -> list[Detection]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3])
        
        labels = np.array(pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points))
        
        detections = []
        for label in np.unique(labels):
            if label == -1:
                continue 
            
            cluster_points = lidar_points[labels == label]
            
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(cluster_points[:, :3]))
            center = bbox.get_center()
            extent = bbox.get_extent()
            
            detection = Detection(
                height=extent[2],
                width=extent[0],
                length=extent[1],
                x=center[0],
                y=center[1],
                z=center[2],
                rotation_y=0.0  # Rotation estimation can be added later
            )
            detections.append(detection)
        
        return detections