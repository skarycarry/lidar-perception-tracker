import os
import numpy as np
from pathlib import Path
from .base import Detection, Detector


def _make_inference_dataset(dataset_cfg, class_names, logger):
    """Create a DatasetTemplate subclass that wraps in-memory point clouds."""
    from pcdet.datasets import DatasetTemplate

    class _InferenceDataset(DatasetTemplate):
        def __len__(self):
            return 0

        def __getitem__(self, i):
            raise NotImplementedError

        def prepare_single(self, points: np.ndarray) -> dict:
            return self.prepare_data(data_dict={'points': points, 'frame_id': 0})

    return _InferenceDataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        training=False,
        root_path=Path('.'),
        logger=logger,
    )


class OpenPCDetDetector(Detector):
    """
    Wraps any OpenPCDet KITTI model (SECOND, PV-RCNN, Voxel R-CNN, etc.).

    Installation:
        pip install spconv-cu124
        git clone https://github.com/open-mmlab/OpenPCDet
        cd OpenPCDet && pip install -r requirements.txt && python setup.py develop

    Args:
        cfg_file:        Path to OpenPCDet model YAML (e.g. tools/cfgs/kitti_models/second.yaml)
        ckpt_path:       Path to pretrained .pth checkpoint
        conf_threshold:  Minimum score to keep a detection
        device:          'cuda' or 'cpu'
    """
    needs_external_preprocessing = False

    def __init__(self, cfg_file: Path, ckpt_path: Path,
                 conf_threshold: float = 0.5, device: str = 'cuda'):
        import torch
        from pcdet.config import cfg as _pcfg, cfg_from_yaml_file
        from pcdet.models import build_network
        from pcdet.utils import common_utils

        self._torch = torch
        self.conf_threshold = conf_threshold

        # cfg paths like _BASE_CONFIG_ are relative to OpenPCDet's tools/ dir
        import pcdet
        openpcdet_tools = Path(pcdet.__file__).parent.parent / 'tools'
        cfg_file_abs = Path(cfg_file).resolve()

        logger = common_utils.create_logger()
        original_cwd = os.getcwd()
        try:
            os.chdir(openpcdet_tools)
            cfg_from_yaml_file(str(cfg_file_abs), _pcfg)
        finally:
            os.chdir(original_cwd)
        self._class_names = list(_pcfg.CLASS_NAMES)

        self._dataset = _make_inference_dataset(
            dataset_cfg=_pcfg.DATA_CONFIG,
            class_names=_pcfg.CLASS_NAMES,
            logger=logger,
        )

        self._model = build_network(
            model_cfg=_pcfg.MODEL,
            num_class=len(_pcfg.CLASS_NAMES),
            dataset=self._dataset,
        )
        self._model.load_params_from_file(filename=str(ckpt_path), to_cpu=True, logger=logger)
        self._dev = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._model.to(self._dev).eval()

    def detect(self, lidar_points: np.ndarray) -> list[Detection]:
        # Import here to handle different OpenPCDet versions
        try:
            from pcdet.models import load_data_to_gpu
        except ImportError:
            from pcdet.utils.common_utils import load_data_to_gpu

        with self._torch.no_grad():
            data_dict = self._dataset.prepare_single(lidar_points)
            batch     = self._dataset.collate_batch([data_dict])
            load_data_to_gpu(batch)
            pred_dicts, _ = self._model.forward(batch)

        pred   = pred_dicts[0]
        boxes  = pred['pred_boxes'].cpu().numpy()   # (N, 7) [x, y, z_ctr, l, w, h, heading]
        scores = pred['pred_scores'].cpu().numpy()
        labels = pred['pred_labels'].cpu().numpy()  # 1-indexed

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if score < self.conf_threshold:
                continue
            x, y, z_ctr, l, w, h, heading = (float(v) for v in box)
            detections.append(Detection(
                x=x, y=y,
                z=z_ctr - h / 2.0,  # OpenPCDet gives center z; Detection.z is bottom
                length=l, width=w, height=h,
                rotation_y=heading,
                confidence=float(score),
                object_type=self._class_names[int(label) - 1],
            ))
        return detections
