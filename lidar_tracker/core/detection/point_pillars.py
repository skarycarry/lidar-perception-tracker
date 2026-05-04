import numpy as np
import torch
from pathlib import Path
from .base import Detection, Detector

class PointPillarsDetector(Detector):
    def __init__(self, model_path: Path, conf_threshold: float = 0.5, nms_threshold: float = 0.5, device: str = 'cuda'):
        self.device = device
        self.model = torch.jit.load(model_path).to(self.device)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def detect(self, lidar_points: np.ndarray) -> list[Detection]:
        tensors = self.preprocess(lidar_points)
        with torch.no_grad():
            outputs = self.model(tensors)
        detections = self.postprocess(outputs)
        return detections
    
    def preprocess(self, lidar_points: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(lidar_points).float().unsqueeze(0).to(self.device)
    
    def postprocess(self, outputs: torch.Tensor) -> list[Detection]:
        detections = []
        for output in outputs:
            if output[0] < self.conf_threshold:
                continue
            detection = Detection(
                height=output[1].item(),
                width=output[2].item(),
                length=output[3].item(),
                x=output[4].item(),
                y=output[5].item(),
                z=output[6].item(),
                rotation_y=output[7].item(),
                object_type=output[8].item(),  # Assuming class index is at position 8
                confidence=output[0].item()
            )
            detections.append(detection)

        # TODO: apply BEV NMS here using self.nms_threshold to remove duplicate detections
        # torchvision.ops.nms(boxes, scores, self.nms_threshold) operates on 2D boxes;
        # for 3D, project to bird's eye view (x, y, l, w) before applying
        return detections