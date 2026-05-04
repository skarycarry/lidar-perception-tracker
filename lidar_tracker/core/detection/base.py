import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Detection:
    height: float
    width: float
    length: float
    x: float
    y: float
    z: float
    rotation_y: float
    confidence: float | None = None
    object_type: str | None = None

class Detector(ABC):
    @abstractmethod
    def detect(self, lidar_points: np.ndarray) -> list[Detection]:
        pass