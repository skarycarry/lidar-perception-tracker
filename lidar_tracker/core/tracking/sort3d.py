import numpy as np
from scipy.optimize import linear_sum_assignment
from .track import Track
from lidar_tracker.core.detection.base import Detection
from lidar_tracker.core.evaluation.iou3d import iou_3d

class Sort3D:
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, detections: list[Detection], dt: float = 1.0) -> list[Track]:
        for track in self.tracks:
            track.predict(dt=dt)

        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Track(initial_detection=det, min_hits=self.min_hits, max_age=self.max_age, track_id=self.next_id))
                self.next_id += 1
        else:
            iou_matrix = self.compute_iou(self.tracks, detections)

            matched_indices = np.array(linear_sum_assignment(-iou_matrix)).T

            unmatched_tracks = set(range(len(self.tracks))) - set(matched_indices[:, 0])
            unmatched_detections = set(range(len(detections))) - set(matched_indices[:, 1])

            for track_idx, det_idx in matched_indices:
                if iou_matrix[track_idx, det_idx] < self.iou_threshold:
                    unmatched_tracks.add(track_idx)
                    unmatched_detections.add(det_idx)
                else:
                    self.tracks[track_idx].update(detections[det_idx])

            for det_idx in unmatched_detections:
                self.tracks.append(Track(initial_detection=detections[det_idx], min_hits=self.min_hits, max_age=self.max_age, track_id=self.next_id))
                self.next_id += 1

            self.tracks = [track for track in self.tracks if not track.is_deleted()]

        return [track for track in self.tracks if track.is_confirmed()]

    def compute_iou(self, tracks: list[Track], detections: list[Detection]) -> np.ndarray:
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = iou_3d(track, det)
        return iou_matrix
