#!/usr/bin/env python3
"""
Sweep tracker + detector confidence parameters for the PointPillars detector.

Detects all frames once per conf_threshold, then re-runs tracking for each
(min_hits, max_age, match_distance) combo — so the expensive GPU pass only
happens once per threshold value.

Usage:
    python scripts/tune_tracker.py --sequence 0000
    python scripts/tune_tracker.py --sequence 0000 --detector euclidean
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_tracker.core.data.kitti_loader import load_calibration, load_labels, KittiDetection
from lidar_tracker.core.tracking.sort3d import Sort3D
from lidar_tracker.core.evaluation.metrics import compute_metrics
from lidar_tracker.core.preprocessing.filters import range_crop, voxel_downsample
from lidar_tracker.core.preprocessing.ground_filter import remove_ground

CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'default.yaml'
KITTI_ROOT  = Path('~/kitti_dataset').expanduser() / 'training'
EVAL_CLASSES = {'Car', 'Van', 'Pedestrian', 'Cyclist'}


def cam_to_velo(points_cam, calib):
    R, T = calib.rect, calib.velo_to_cam
    R_full = R @ T[:, :3]
    t_full = R @ T[:, 3]
    R_inv  = R_full.T
    return (R_inv @ points_cam.T).T + (-R_inv @ t_full)


def gt_to_lidar(det, calib):
    center_cam  = np.array([[det.x, det.y - det.height / 2.0, det.z]])
    center_velo = cam_to_velo(center_cam, calib)[0]
    return KittiDetection(
        track_id=det.track_id, object_type=det.object_type,
        height=det.height, width=det.width, length=det.length,
        x=float(center_velo[0]), y=float(center_velo[1]),
        z=float(center_velo[2] - det.height / 2.0),
        rotation_y=-det.rotation_y,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', default='0000')
    parser.add_argument('--detector', choices=['point_pillars', 'euclidean'], default='point_pillars')
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text())
    seq    = args.sequence

    calib      = load_calibration(KITTI_ROOT / 'calib'    / f'{seq}.txt')
    raw_labels = load_labels    (KITTI_ROOT / 'label_02' / f'{seq}.txt')
    gt_lidar   = {}
    for fid, dets in raw_labels.items():
        filtered = [gt_to_lidar(d, calib) for d in dets if d.object_type in EVAL_CLASSES]
        if filtered:
            gt_lidar[fid] = filtered

    frame_files = sorted((KITTI_ROOT / 'velodyne' / seq).glob('*.bin'))
    print(f'Sequence {seq}: {len(frame_files)} frames, {len(gt_lidar)} GT frames\n')

    # ── Tracker search grid ────────────────────────────────────────────────────
    min_hits_vals    = [1, 2, 3]
    max_age_vals     = [1, 3, 5, 8]
    match_dist_vals  = [1.0, 2.0, 3.0]

    if args.detector == 'point_pillars':
        conf_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
        pre_cfg = config['preprocessing']

        from lidar_tracker.core.detection.point_pillars import PointPillarsDetector
        model_path   = Path(config['detection']['point_pillars']['model_path'])
        nms_threshold = config['detection']['point_pillars']['nms_threshold']
        device       = config['detection']['point_pillars']['device']

        best_mota, best_cfg = -float('inf'), {}

        for conf in conf_thresholds:
            det = PointPillarsDetector(model_path, conf_threshold=conf,
                                       nms_threshold=nms_threshold, device=device)
            print(f'Detecting with conf={conf:.1f}...')
            frame_dets = []
            for f in frame_files:
                pts = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
                frame_dets.append(det.detect(pts))

            for min_hits in min_hits_vals:
                for max_age in max_age_vals:
                    for match_dist in match_dist_vals:
                        trk = Sort3D(max_age=max_age, min_hits=min_hits,
                                     match_distance=match_dist)
                        all_tracks = {}
                        for fi, dets in enumerate(frame_dets):
                            all_tracks[fi] = list(trk.update(dets, dt=0.1))
                        m = compute_metrics(gt_lidar, all_tracks,
                                            match_distance=config['evaluation']['match_distance'])
                        mota = m['MOTA']
                        flag = ''
                        if mota > best_mota:
                            best_mota = mota
                            best_cfg  = dict(conf=conf, min_hits=min_hits,
                                             max_age=max_age, match_dist=match_dist)
                            flag = ' ◀ best'
                        print(f'  conf={conf:.1f} min_hits={min_hits} max_age={max_age:2d} '
                              f'mdist={match_dist:.1f}  '
                              f'MOTA={mota:+.4f}  '
                              f'matches={m["Total Matches"]:3d}  '
                              f'FP={m["False Positives"]:4d}  '
                              f'FN={m["False Negatives"]:3d}{flag}')

    else:  # euclidean — no conf threshold to sweep
        pre_cfg = config['preprocessing']
        from lidar_tracker.core.detection.euclidean import EuclideanDetector
        ecfg = config['detection']['euclidean']
        det  = EuclideanDetector(
            eps=ecfg['eps'], min_points=ecfg['min_points'],
            min_h=ecfg['min_h'], max_h=ecfg['max_h'],
            min_w=ecfg['min_w'], max_w=ecfg['max_w'],
            min_l=ecfg['min_l'], max_l=ecfg['max_l'],
            max_center_z=ecfg['max_center_z'],
        )
        print('Detecting with euclidean...')
        frame_dets = []
        for f in frame_files:
            pts = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
            pts = range_crop(pts, pre_cfg['min_distance'], pre_cfg['max_distance'])
            pts = remove_ground(pts, pre_cfg['ground_threshold'])
            pts = voxel_downsample(pts, pre_cfg['voxel_size'])
            frame_dets.append(det.detect(pts))

        best_mota, best_cfg = -float('inf'), {}
        for min_hits in min_hits_vals:
            for max_age in max_age_vals:
                for match_dist in match_dist_vals:
                    trk = Sort3D(max_age=max_age, min_hits=min_hits,
                                 match_distance=match_dist)
                    all_tracks = {}
                    for fi, dets in enumerate(frame_dets):
                        all_tracks[fi] = list(trk.update(dets, dt=0.1))
                    m = compute_metrics(gt_lidar, all_tracks,
                                        match_distance=config['evaluation']['match_distance'])
                    mota = m['MOTA']
                    flag = ''
                    if mota > best_mota:
                        best_mota = mota
                        best_cfg  = dict(min_hits=min_hits, max_age=max_age,
                                         match_dist=match_dist)
                        flag = ' ◀ best'
                    print(f'  min_hits={min_hits} max_age={max_age:2d} '
                          f'mdist={match_dist:.1f}  '
                          f'MOTA={mota:+.4f}  '
                          f'matches={m["Total Matches"]:3d}  '
                          f'FP={m["False Positives"]:4d}  '
                          f'FN={m["False Negatives"]:3d}{flag}')

    print(f'\nBest MOTA: {best_mota:+.4f}')
    print('Best config:', best_cfg)


if __name__ == '__main__':
    main()
