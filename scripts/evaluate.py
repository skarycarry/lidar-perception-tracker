#!/usr/bin/env python3
"""
Offline evaluation script for a single KITTI tracking sequence.
Transforms GT labels from camera to LiDAR frame, runs detector + tracker,
and reports MOTA/MOTP metrics.

Usage:
    python scripts/evaluate.py --sequence 0000
"""
import argparse
import numpy as np
from pathlib import Path

from lidar_tracker.core.data.kitti_loader import (
    load_lidar_frames, load_calibration, load_labels, KittiDetection
)
from lidar_tracker.core.detection.factory import create_detector
from lidar_tracker.core.tracking.sort3d import Sort3D
from lidar_tracker.core.evaluation.metrics import compute_metrics
from lidar_tracker.core.preprocessing.filters import range_crop, voxel_downsample
from lidar_tracker.core.preprocessing.ground_filter import remove_ground
import yaml

EVAL_CLASSES = {'Car', 'Van', 'Pedestrian', 'Cyclist'}
CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'default.yaml'
KITTI_ROOT = Path('~/kitti_dataset').expanduser() / 'training'


def cam_to_velo(points_cam: np.ndarray, calib) -> np.ndarray:
    """Transform Nx3 points from rectified camera frame to LiDAR frame."""
    R = calib.rect                          # 3x3 rectification
    T = calib.velo_to_cam                   # 3x4 [R|t]
    R_vc = T[:, :3]                         # 3x3 rotation velo->cam
    t_vc = T[:, 3]                          # 3   translation

    # full 3x3 from velo to rectified cam
    R_full = R @ R_vc
    t_full = R @ t_vc

    # invert: cam_rect -> velo
    R_inv = R_full.T
    t_inv = -R_inv @ t_full

    return (R_inv @ points_cam.T).T + t_inv


def gt_to_lidar(det: KittiDetection, calib) -> KittiDetection:
    """Return a new KittiDetection with x,y,z in LiDAR frame."""
    # KITTI y is at bottom of box in camera frame; center is y - h/2
    center_cam = np.array([[det.x, det.y - det.height / 2.0, det.z]])
    center_velo = cam_to_velo(center_cam, calib)[0]
    return KittiDetection(
        track_id=det.track_id,
        object_type=det.object_type,
        height=det.height,
        width=det.width,
        length=det.length,
        x=float(center_velo[0]),
        y=float(center_velo[1]),
        z=float(center_velo[2] - det.height / 2.0),  # bottom of box in LiDAR frame
        rotation_y=-det.rotation_y,  # camera Y-down → LiDAR Z-up flips sign
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', default='0000', help='KITTI sequence ID e.g. 0000')
    args = parser.parse_args()

    seq = args.sequence
    velodyne_dir = KITTI_ROOT / 'velodyne' / seq
    calib_file   = KITTI_ROOT / 'calib'    / f'{seq}.txt'
    label_file   = KITTI_ROOT / 'label_02' / f'{seq}.txt'

    print(f'Evaluating sequence {seq}')

    calib = load_calibration(calib_file)
    raw_labels = load_labels(label_file)

    # Filter to eval classes and transform GT to LiDAR frame
    gt_lidar: dict[int, list[KittiDetection]] = {}
    for frame_id, dets in raw_labels.items():
        filtered = [gt_to_lidar(d, calib) for d in dets if d.object_type in EVAL_CLASSES]
        if filtered:
            gt_lidar[frame_id] = filtered

    config = yaml.safe_load(CONFIG_PATH.read_text())
    pre_cfg = config['preprocessing']

    detector = create_detector(CONFIG_PATH)
    trk_cfg = config['tracking']
    tracker = Sort3D(
        max_age=trk_cfg['max_age'],
        min_hits=trk_cfg['min_hits'],
        match_distance=trk_cfg['match_distance'],
    )

    all_tracks: dict[int, list] = {}
    all_detections: dict[int, list] = {}
    frame_files = sorted(velodyne_dir.glob('*.bin'))

    print(f'Processing {len(frame_files)} frames...')
    for frame_idx, bin_file in enumerate(frame_files):
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        points = range_crop(points, pre_cfg['min_distance'], pre_cfg['max_distance'])
        points = remove_ground(points, pre_cfg['ground_threshold'])
        points = voxel_downsample(points, pre_cfg['voxel_size'])
        detections = detector.detect(points)
        tracks = tracker.update(detections, dt=0.1)
        all_tracks[frame_idx] = list(tracks)
        all_detections[frame_idx] = detections

    from lidar_tracker.core.evaluation.geometry3d import iou_3d

    def _center(b) -> np.ndarray:
        return np.array([b.x, b.y, b.z + b.height / 2.0])

    def _nearest_det(gt, dets):
        if not dets:
            return None
        return min(dets, key=lambda d: float(np.linalg.norm(_center(gt) - _center(d))))

    def _best_iou_det(gt, dets):
        if not dets:
            return None, 0.0
        best_d, best_v = None, 0.0
        for d in dets:
            v = iou_3d(gt, d)
            if v > best_v:
                best_d, best_v = d, v
        return best_d, best_v

    # --- Diagnostics ---
    det_counts = [len(v) for v in all_detections.values()]
    print(f'\n--- Diagnostics ---')
    print(f'  Detections per frame: min={min(det_counts)} max={max(det_counts)} mean={np.mean(det_counts):.1f}')

    # Frame 0: per-GT nearest detection and best-IoU detection (vs raw detections, not tracks)
    if 0 in gt_lidar and 0 in all_detections:
        dets0 = all_detections[0]
        print(f'\n  Frame 0 per-GT match analysis ({len(dets0)} raw detections):')
        hdr = f'  {"":12s}  {"x":>6}  {"y":>6}  {"z":>6}  {"h":>5}  {"w":>5}  {"l":>5}  {"ry":>7}'
        for g in gt_lidar[0]:
            nearest = _nearest_det(g, dets0)
            best_d, best_v = _best_iou_det(g, dets0)
            dist = float(np.linalg.norm(_center(g) - _center(nearest))) if nearest else float('inf')
            print(f'\n    GT  [{g.object_type}]')
            print(hdr)
            print(f'  {"  GT":12s}  {g.x:6.2f}  {g.y:6.2f}  {g.z:6.2f}  '
                  f'{g.height:5.2f}  {g.width:5.2f}  {g.length:5.2f}  {g.rotation_y:7.3f}')
            if nearest:
                print(f'  {"  Nearest":12s}  {nearest.x:6.2f}  {nearest.y:6.2f}  {nearest.z:6.2f}  '
                      f'{nearest.height:5.2f}  {nearest.width:5.2f}  {nearest.length:5.2f}  '
                      f'{nearest.rotation_y:7.3f}  dist={dist:.2f}m  iou={iou_3d(g, nearest):.4f}')
            if best_d is not None and best_d is not nearest:
                print(f'  {"  Best IoU":12s}  {best_d.x:6.2f}  {best_d.y:6.2f}  {best_d.z:6.2f}  '
                      f'{best_d.height:5.2f}  {best_d.width:5.2f}  {best_d.length:5.2f}  '
                      f'{best_d.rotation_y:7.3f}  iou={best_v:.4f}')
            elif best_d is not None:
                print(f'  (nearest == best IoU detection, iou={best_v:.4f})')

    # Best center distance per GT vs raw detections across all frames
    match_dist = config['evaluation']['match_distance']
    best_dists = []
    for frame_id, gt_dets in gt_lidar.items():
        dets = all_detections.get(frame_id, [])
        for gt in gt_dets:
            if not dets:
                best_dists.append(float('inf'))
                continue
            d = min(float(np.linalg.norm(_center(gt) - _center(det))) for det in dets)
            best_dists.append(d)
    finite = [d for d in best_dists if np.isfinite(d)]
    if finite:
        print(f'\n  Best detection center distance per GT (raw detections, all frames):')
        print(f'    min={min(finite):.3f}m  max={max(finite):.3f}m  mean={np.mean(finite):.3f}m')
        print(f'    <= 0.5m: {sum(1 for d in finite if d <= 0.5)} / {len(best_dists)}')
        print(f'    <= 1.0m: {sum(1 for d in finite if d <= 1.0)} / {len(best_dists)}')
        print(f'    <= {match_dist}m: {sum(1 for d in finite if d <= match_dist)} / {len(best_dists)}')

    metrics = compute_metrics(gt_lidar, all_tracks, match_distance=match_dist)

    print('\n--- Evaluation Results ---')
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
