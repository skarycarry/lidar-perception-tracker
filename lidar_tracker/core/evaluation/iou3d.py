from lidar_tracker.core.detection import Detection
from lidar_tracker.core.tracking import Track

def iou_3d(track: Track, detection: Detection) -> float:
    track_detection = track.last_detection
    track_x_min, track_x_max, track_y_min, track_y_max, track_z_min, track_z_max = track_detection.bounds
    detection_x_min, detection_x_max, detection_y_min, detection_y_max, detection_z_min, detection_z_max = detection.bounds

    x_overlap = max(0, min(track_x_max, detection_x_max) - max(track_x_min, detection_x_min))
    y_overlap = max(0, min(track_y_max, detection_y_max) - max(track_y_min, detection_y_min))
    z_overlap = max(0, min(track_z_max, detection_z_max) - max(track_z_min, detection_z_min))

    intersection_volume = x_overlap * y_overlap * z_overlap
    track_volume = track_detection.volume
    detection_volume = detection.volume
    union_volume = track_volume + detection_volume - intersection_volume
    if union_volume == 0:
        return 0.0
    return intersection_volume / union_volume