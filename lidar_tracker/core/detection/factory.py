import yaml
from pathlib import Path
from .base import Detector
from .euclidean import EuclideanDetector
from .point_pillars import PointPillarsDetector

def create_detector(config_path: Path) -> Detector:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config['detection']['mode'] == 'euclidean':
        eps = config['detection']['euclidean']['eps']
        min_points = config['detection']['euclidean']['min_points']
        return EuclideanDetector(eps=eps, min_points=min_points)
    elif config['detection']['mode'] == 'point_pillars':
        model_path = Path(config['detection']['point_pillars']['model_path'])
        conf_threshold = config['detection']['point_pillars']['conf_threshold']
        nms_threshold = config['detection']['point_pillars']['nms_threshold']
        device = config['detection']['point_pillars']['device']
        return PointPillarsDetector(model_path=model_path, conf_threshold=conf_threshold, nms_threshold=nms_threshold, device=device)
    else:
        raise ValueError(f"Unknown detector type: {config['detection']['mode']}")