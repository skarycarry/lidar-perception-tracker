#!/usr/bin/env python3
"""
Parameter search for the Euclidean detector using random exploration + hill-climbing.

Strategy:
  - First half of iterations: pure random sampling (explore the full space)
  - Second half: Gaussian perturbation around the best params found so far (exploit)
  This mirrors how NES/evolution strategies work for non-differentiable objectives.

Results are streamed to a CSV so you can inspect them while the search runs.

Usage:
    python scripts/tune_detector.py --sequence 0000 --iterations 100
    python scripts/tune_detector.py --sequence 0000 --iterations 200 --fast
"""
import argparse
import csv
import random
import time
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_tracker.core.data.kitti_loader import load_calibration, load_labels, KittiDetection
from lidar_tracker.core.detection.euclidean import EuclideanDetector
from lidar_tracker.core.tracking.sort3d import Sort3D
from lidar_tracker.core.evaluation.metrics import compute_metrics
from lidar_tracker.core.preprocessing.filters import range_crop, voxel_downsample
from lidar_tracker.core.preprocessing.ground_filter import remove_ground

CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'default.yaml'
KITTI_ROOT = Path('~/kitti_dataset').expanduser() / 'training'
EVAL_CLASSES = {'Car', 'Van', 'Pedestrian', 'Cyclist'}

# (low, high) for each parameter — int ranges use randint, float ranges use uniform
PARAM_RANGES = {
    'eps':          (0.3,  1.5,  'float'),
    'min_points':   (5,    60,   'int'),
    'min_h':        (0.3,  1.0,  'float'),
    'max_h':        (1.5,  5.0,  'float'),
    'min_w':        (0.2,  0.8,  'float'),
    'max_w':        (0.8,  4.0,  'float'),
    'min_l':        (0.3,  2.0,  'float'),
    'max_l':        (2.0,  8.0,  'float'),
    'max_center_z': (-0.5, 2.0,  'float'),
}


def _cam_to_velo(points_cam, calib):
    R, T = calib.rect, calib.velo_to_cam
    R_full = R @ T[:, :3]
    t_full = R @ T[:, 3]
    R_inv = R_full.T
    return (R_inv @ points_cam.T).T + (-R_inv @ t_full)


def _gt_to_lidar(det, calib):
    center_cam = np.array([[det.x, det.y - det.height / 2.0, det.z]])
    cv = _cam_to_velo(center_cam, calib)[0]
    return KittiDetection(
        track_id=det.track_id, object_type=det.object_type,
        height=det.height, width=det.width, length=det.length,
        x=float(cv[0]), y=float(cv[1]),
        z=float(cv[2] - det.height / 2.0),
        rotation_y=-det.rotation_y,
    )


def _sample(ranges: dict) -> dict:
    p = {}
    for k, (lo, hi, typ) in ranges.items():
        p[k] = random.randint(int(lo), int(hi)) if typ == 'int' else random.uniform(lo, hi)
    return _fix(p)


def _perturb(params: dict, ranges: dict, scale: float = 0.15) -> dict:
    p = {}
    for k, (lo, hi, typ) in ranges.items():
        span = hi - lo
        delta = random.gauss(0, scale * span)
        raw = params[k] + delta
        p[k] = int(np.clip(round(raw), lo, hi)) if typ == 'int' else float(np.clip(raw, lo, hi))
    return _fix(p)


def _fix(p: dict) -> dict:
    if p['min_h'] > p['max_h']:
        p['min_h'], p['max_h'] = p['max_h'], p['min_h']
    if p['min_w'] > p['max_w']:
        p['min_w'], p['max_w'] = p['max_w'], p['min_w']
    if p['min_l'] > p['max_l']:
        p['min_l'], p['max_l'] = p['max_l'], p['min_l']
    return p


def _evaluate(params, gt_lidar, frame_files, pre_cfg, match_distance, frame_step):
    detector = EuclideanDetector(
        eps=params['eps'], min_points=params['min_points'],
        min_h=params['min_h'], max_h=params['max_h'],
        min_w=params['min_w'], max_w=params['max_w'],
        min_l=params['min_l'], max_l=params['max_l'],
        max_center_z=params['max_center_z'],
    )
    tracker = Sort3D(max_age=5, min_hits=3, match_distance=match_distance)
    all_tracks = {}

    for frame_idx in range(0, len(frame_files), frame_step):
        points = np.fromfile(frame_files[frame_idx], dtype=np.float32).reshape(-1, 4)
        points = range_crop(points, pre_cfg['min_distance'], pre_cfg['max_distance'])
        points = remove_ground(points, pre_cfg['ground_threshold'])
        points = voxel_downsample(points, pre_cfg['voxel_size'])
        detections = detector.detect(points)
        tracks = tracker.update(detections, dt=0.1)
        all_tracks[frame_idx] = list(tracks)

    gt_subset = {k: v for k, v in gt_lidar.items() if k % frame_step == 0}
    return compute_metrics(gt_subset, all_tracks, match_distance=match_distance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', default='0000')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--fast', action='store_true',
                        help='Evaluate on every 5th frame (~5x faster, noisier objective)')
    parser.add_argument('--output', default='tune_results.csv')
    args = parser.parse_args()

    frame_step = 5 if args.fast else 1
    config = yaml.safe_load(CONFIG_PATH.read_text())
    pre_cfg = config['preprocessing']
    match_distance = config['evaluation']['match_distance']

    seq = args.sequence
    calib = load_calibration(KITTI_ROOT / 'calib' / f'{seq}.txt')
    raw_labels = load_labels(KITTI_ROOT / 'label_02' / f'{seq}.txt')

    gt_lidar = {}
    for frame_id, dets in raw_labels.items():
        filtered = [_gt_to_lidar(d, calib) for d in dets if d.object_type in EVAL_CLASSES]
        if filtered:
            gt_lidar[frame_id] = filtered

    frame_files = sorted((KITTI_ROOT / 'velodyne' / seq).glob('*.bin'))
    print(f'Sequence {seq}: {len(frame_files)} frames, '
          f'{"every " + str(frame_step) + "th" if frame_step > 1 else "all"} evaluated per iteration')
    print(f'Running {args.iterations} iterations → results in {args.output}\n')

    best_mota = -float('inf')
    best_params = None
    explore_iters = args.iterations // 2

    fieldnames = list(PARAM_RANGES.keys()) + ['MOTA', 'MOTP_m', 'FP', 'FN', 'Matches', 'IDSw', 'F1', 's']
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(args.iterations):
            phase = 'explore' if (i < explore_iters or best_params is None) else 'exploit'
            params = _sample(PARAM_RANGES) if phase == 'explore' else _perturb(best_params, PARAM_RANGES)

            t0 = time.time()
            try:
                m = _evaluate(params, gt_lidar, frame_files, pre_cfg, match_distance, frame_step)
            except Exception as e:
                print(f'  [{i+1:3d}] ERROR: {e}')
                continue
            elapsed = time.time() - t0

            matches = m['Total Matches']
            fp = m['False Positives']
            fn = m['False Negatives']
            # F1 over tracking output: no degenerate "detect nothing" optimum
            f1 = 2 * matches / (2 * matches + fp + fn) if (matches + fp + fn) > 0 else 0.0

            if f1 > best_mota:
                best_mota = f1
                best_params = params
                flag = ' ◀ best'
            else:
                flag = ''

            writer.writerow({
                **{k: params[k] for k in PARAM_RANGES},
                'MOTA': round(m['MOTA'], 4),
                'MOTP_m': round(m['MOTP (mean center dist m)'], 4),
                'FP': fp,
                'FN': fn,
                'Matches': matches,
                'IDSw': m['ID Switches'],
                'F1': round(f1, 4),
                's': round(elapsed, 1),
            })
            f.flush()

            print(f'  [{i+1:3d}/{args.iterations}] {phase:7s}  '
                  f'F1={f1:.4f}  MOTA={m["MOTA"]:+.4f}  '
                  f'FP={fp:4d}  FN={fn:3d}  matches={matches:3d}  '
                  f'eps={params["eps"]:.2f}  min_pts={params["min_points"]:2d}  '
                  f'max_cz={params["max_center_z"]:.2f}  [{elapsed:.0f}s]{flag}')

    print(f'\nSearch complete.')
    print(f'Best F1: {best_mota:.4f}')
    print(f'Best params:')
    for k, v in best_params.items():
        print(f'  {k}: {v:.3f}' if isinstance(v, float) else f'  {k}: {v}')


if __name__ == '__main__':
    main()
