import numpy as np
from .base import Detection, Detector


def _center(d: Detection) -> np.ndarray:
    return np.array([d.x, d.y, d.z + d.height / 2.0])


def _nms(detections: list[Detection], nms_distance: float) -> list[Detection]:
    """Greedy center-distance NMS applied per class."""
    if not detections:
        return []

    by_class: dict[str, list[Detection]] = {}
    for d in detections:
        by_class.setdefault(d.object_type or '', []).append(d)

    kept = []
    for cls_dets in by_class.values():
        sorted_dets = sorted(cls_dets, key=lambda d: d.confidence or 0.0, reverse=True)
        suppressed = [False] * len(sorted_dets)
        for i, det_i in enumerate(sorted_dets):
            if suppressed[i]:
                continue
            kept.append(det_i)
            ci = _center(det_i)
            for j in range(i + 1, len(sorted_dets)):
                if not suppressed[j] and np.linalg.norm(ci - _center(sorted_dets[j])) < nms_distance:
                    suppressed[j] = True
    return kept


class FusionDetector(Detector):
    """
    Runs multiple detectors on the same point cloud and merges results with
    greedy center-distance NMS per class.

    Higher-confidence detections take priority; any detection within
    nms_distance of a kept detection (same class) is suppressed.
    """
    needs_external_preprocessing = False

    def __init__(self, detectors: list[Detector], nms_distance: float = 1.5):
        self._detectors = detectors
        self._nms_distance = nms_distance

    def detect(self, lidar_points: np.ndarray) -> list[Detection]:
        all_dets: list[Detection] = []
        for detector in self._detectors:
            all_dets.extend(detector.detect(lidar_points))
        return _nms(all_dets, self._nms_distance)
