import numpy as np
from scipy.optimize import linear_sum_assignment
from lidar_tracker.core.data.kitti_loader import KittiDetection
from lidar_tracker.core.tracking.track import Track


def _center(box) -> np.ndarray:
    return np.array([box.x, box.y, box.z + box.height / 2.0])


def compute_metrics(
    all_gt: dict[int, list[KittiDetection]],
    all_tracks: dict[int, list[Track]],
    match_distance: float = 2.0,
) -> dict[str, float]:
    prev_matches = {}
    total_gt = 0
    total_matches = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_id_switches = 0
    total_dist = 0.0

    for frame, gt_dets in all_gt.items():
        tracks = all_tracks.get(frame, [])
        total_gt += len(gt_dets)

        dist_matrix = np.full((len(gt_dets), len(tracks)), np.inf)
        for i, gt in enumerate(gt_dets):
            for j, track in enumerate(tracks):
                dist_matrix[i, j] = float(np.linalg.norm(_center(gt) - _center(track.last_detection)))

        raw = np.array(linear_sum_assignment(dist_matrix)).T
        matched_indices = [(i, j) for i, j in raw if dist_matrix[i, j] <= match_distance]

        matched_gt = set()
        matched_tracks = set()
        for gt_idx, track_idx in matched_indices:
            matched_gt.add(gt_idx)
            matched_tracks.add(track_idx)
            total_dist += dist_matrix[gt_idx, track_idx]
            if frame - 1 in prev_matches and gt_idx in prev_matches[frame - 1]:
                if prev_matches[frame - 1][gt_idx] != tracks[track_idx].track_id:
                    total_id_switches += 1

        total_matches += len(matched_gt)
        total_false_positives += len(tracks) - len(matched_tracks)
        total_false_negatives += len(gt_dets) - len(matched_gt)
        prev_matches[frame] = {gt_idx: tracks[track_idx].track_id for gt_idx, track_idx in matched_indices}

    mota = 1 - (total_false_positives + total_false_negatives + total_id_switches) / total_gt if total_gt > 0 else 0.0
    motp = total_dist / total_matches if total_matches > 0 else 0.0
    return {
        "MOTA": mota,
        "MOTP (mean center dist m)": motp,
        "ID Switches": total_id_switches,
        "False Positives": total_false_positives,
        "False Negatives": total_false_negatives,
        "Total GT": total_gt,
        "Total Matches": total_matches,
    }
