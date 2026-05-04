import numpy as np
from dataclasses import dataclass
from .kitti_loader import KittiDetection

@dataclass
class SyntheticScene:
    width: int
    height: int
    objects: list[KittiDetection]

def generate_synthetic_scene(num_objects: int, scene_width: int, scene_height: int) -> SyntheticScene:
    objects = []
    for i in range(num_objects):
        object_type = np.random.choice(['Car', 'Pedestrian', 'Cyclist'])
        height = np.random.uniform(1.0, 3.0)
        width = np.random.uniform(0.5, 2.0)
        length = np.random.uniform(1.0, 4.0)
        x = np.random.uniform(-scene_width / 2, scene_width / 2)
        y = np.random.uniform(-scene_height / 2, scene_height / 2)
        z = np.random.uniform(0, 10)
        objects.append(KittiDetection(track_id=i, object_type=object_type, height=height, width=width, length=length, x=x, y=y, z=z, rotation_y=0.0))
    
    return SyntheticScene(width=scene_width, height=scene_height, objects=objects)

def generate_synthetic_lidar_points(scene: SyntheticScene, num_points: int) -> tuple[np.ndarray, dict[int, list[KittiDetection]]]:
    points = np.random.uniform(low=[-scene.width / 2, -scene.height / 2, 0, 0], high=[scene.width / 2, scene.height / 2, 10, 1], size=(num_points, 4))
    for obj in scene.objects:
        for _ in range(num_points // len(scene.objects)):
            x = obj.x + np.random.uniform(-obj.width / 2, obj.width / 2)
            y = obj.y + np.random.uniform(-obj.length / 2, obj.length / 2)
            z = obj.z + np.random.uniform(0, obj.height)
            points = np.vstack([points, [x, y, z, 1.0]])
    return points, {0: scene.objects}