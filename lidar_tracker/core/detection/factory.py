import yaml
from pathlib import Path
from .base import Detector

def create_detector(config_path: Path) -> Detector:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    mode = config['detection']['mode']
    if mode == 'euclidean':
        from .euclidean import EuclideanDetector
        ecfg = config['detection']['euclidean']
        return EuclideanDetector(
            eps=ecfg['eps'],
            min_points=ecfg['min_points'],
            min_h=ecfg['min_h'], max_h=ecfg['max_h'],
            min_w=ecfg['min_w'], max_w=ecfg['max_w'],
            min_l=ecfg['min_l'], max_l=ecfg['max_l'],
            max_center_z=ecfg['max_center_z'],
        )
    elif mode == 'point_pillars':
        from .point_pillars import PointPillarsDetector
        cfg = config['detection']['point_pillars']
        return PointPillarsDetector(
            model_path=Path(cfg['model_path']),
            conf_threshold=cfg['conf_threshold'],
            nms_threshold=cfg['nms_threshold'],
            device=cfg['device'],
        )
    elif mode in ('second', 'pvrcnn', 'voxel_rcnn'):
        from .openpcdet_detector import OpenPCDetDetector
        cfg = config['detection'][mode]
        return OpenPCDetDetector(
            cfg_file=Path(cfg['cfg_file']),
            ckpt_path=Path(cfg['model_path']),
            conf_threshold=cfg['conf_threshold'],
            device=cfg['device'],
        )
    elif mode == 'fusion':
        from .point_pillars import PointPillarsDetector
        from .openpcdet_detector import OpenPCDetDetector
        from .fusion import FusionDetector
        pp_cfg = config['detection']['point_pillars']
        pv_cfg = config['detection']['pvrcnn']
        fu_cfg = config['detection']['fusion']
        pp = PointPillarsDetector(
            model_path=Path(pp_cfg['model_path']),
            conf_threshold=pp_cfg['conf_threshold'],
            nms_threshold=pp_cfg['nms_threshold'],
            device=pp_cfg['device'],
        )
        pv = OpenPCDetDetector(
            cfg_file=Path(pv_cfg['cfg_file']),
            ckpt_path=Path(pv_cfg['model_path']),
            conf_threshold=pv_cfg['conf_threshold'],
            device=pv_cfg['device'],
        )
        return FusionDetector([pp, pv], nms_distance=fu_cfg['nms_distance'])
    else:
        raise ValueError(f"Unknown detector mode: {mode}")
