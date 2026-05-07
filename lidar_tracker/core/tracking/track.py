from .kalman import KalmanFilter3D
from lidar_tracker.core.detection.base import Detection

class Track:
    def __init__(self, initial_detection: Detection, min_hits=3, max_age=5, track_id=None):
        self.kalman_filter = KalmanFilter3D(initial_detection.initial_track_state)
        self.last_detection = initial_detection
        self.track_id = track_id
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.min_hits = min_hits
        self.max_age = max_age

    @property
    def state(self):
        return self.kalman_filter.state

    def predict(self, dt):
        self.kalman_filter.predict(dt)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection):
        self.kalman_filter.update(detection.position)
        self.last_detection = detection
        self.hits += 1
        self.time_since_update = 0

    def is_confirmed(self):
        return self.hits >= self.min_hits
    
    def is_deleted(self):
        return self.time_since_update > self.max_age