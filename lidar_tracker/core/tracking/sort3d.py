import numpy as np
from scipy.optimize import linear_sum_assignment
from .track import Track
from lidar_tracker.core.detection.base import Detection

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
                self.tracks.append(Track(initial_state=det.initial_track_state, min_hits=self.min_hits, max_age=self.max_age, track_id=self.next_id))
                self.next_id += 1
        else:
            track_states = np.array([track.state for track in self.tracks])
            iou_matrix = self.compute_iou(track_states, detections)

            matched_indices = np.array(linear_sum_assignment(-iou_matrix)).T

            unmatched_tracks = set(range(len(self.tracks))) - set(matched_indices[:, 0])
            unmatched_detections = set(range(len(detections))) - set(matched_indices[:, 1])

            for track_idx, det_idx in matched_indices:
                if iou_matrix[track_idx, det_idx] < self.iou_threshold:
                    unmatched_tracks.add(track_idx)
                    unmatched_detections.add(det_idx)
                else:
                    self.tracks[track_idx].update(detections[det_idx].position)

            for det_idx in unmatched_detections:
                self.tracks.append(Track(initial_state=detections[det_idx].initial_track_state, min_hits=self.min_hits, max_age=self.max_age, track_id=self.next_id))
                self.next_id += 1

            self.tracks = [track for track in self.tracks if not track.is_deleted()]

        return [track for track in self.tracks if track.is_confirmed()]

    def compute_iou(self, track_states: np.ndarray, detections: list[Detection]) -> np.ndarray:
        iou_matrix = np.zeros((len(track_states), len(detections)))
        for i, track_state in enumerate(track_states):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self.iou_3d(track_state, det)
        return iou_matrix

    def iou_3d(self, track_state: np.ndarray, _detection: Detection) -> float:
        return 0.0
