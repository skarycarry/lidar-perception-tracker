import numpy as np
from pathlib import Path
from collections.abc import Iterator
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class KittiCalibration:
    velo_to_cam: np.ndarray
    rect: np.ndarray
    proj_matrix: np.ndarray

@dataclass
class KittiDetection:
    track_id: int
    object_type: str
    height: float
    width: float
    length: float
    x: float
    y: float
    z: float
    rotation_y: float

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.x - self.length / 2, self.x + self.length / 2,
            self.y - self.width / 2,  self.y + self.width / 2,
            self.z,                   self.z + self.height,
        )

    @property
    def volume(self) -> float:
        return self.height * self.width * self.length


def load_lidar_frames(dir_path: Path) -> Iterator[np.ndarray]:
    for file in sorted(dir_path.iterdir()):
        if file.suffix == '.bin':
            lidar_points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
            yield lidar_points

def load_calibration(calib_file: Path) -> KittiCalibration:
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    
    calib_data = {}
    for line in lines:
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        calib_data[key.strip()] = np.array([float(x) for x in value.strip().split()])
    
    velo_to_cam = calib_data['Tr_velo_to_cam'].reshape(3, 4)
    rect = calib_data['R0_rect'].reshape(3, 3)
    proj_matrix = calib_data['P2'].reshape(3, 4)
    
    return KittiCalibration(velo_to_cam=velo_to_cam, rect=rect, proj_matrix=proj_matrix)

def load_labels(label_file: Path) -> dict[int, list[KittiDetection]]:
    detections = defaultdict(list)
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 17:
            continue
        
        frame_id = int(parts[0])
        track_id = int(parts[1])
        object_type = parts[2]
        height = float(parts[10])
        width = float(parts[11])
        length = float(parts[12])
        x = float(parts[13])
        y = float(parts[14])
        z = float(parts[15])
        rotation_y = float(parts[16])
        
        detections[frame_id].append(KittiDetection(
            track_id=track_id,
            object_type=object_type,
            height=height,
            width=width,
            length=length,
            x=x,
            y=y,
            z=z,
            rotation_y=rotation_y
        ))

    return detections
