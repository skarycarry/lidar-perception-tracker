#!/usr/bin/env python3
"""
Sweep tracker + detector parameters to maximise MOTA.

Ego motion is precomputed once per sequence so every tracker combo sees
world-frame detections — matching the live pipeline exactly.  Tracks are
converted back to sensor frame before metrics comparison with KITTI GT.

Usage:
    python scripts/tune_tracker.py
    python scripts/tune_tracker.py --sequences 0000 0001 0002
    python scripts/tune_tracker.py --detector euclidean
    python scripts/tune_tracker.py --write-config   # save best params to default.yaml
"""
import argparse
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_tracker.core.data.kitti_loader import load_calibration, load_labels, KittiDetection
from lidar_tracker.core.detection.base import Detection
from lidar_tracker.core.tracking.sort3d import Sort3D
from lidar_tracker.core.evaluation.metrics import compute_metrics
from lidar_tracker.core.preprocessing.filters import range_crop, voxel_downsample
from lidar_tracker.core.preprocessing.ground_filter import remove_ground
from lidar_tracker.core.preprocessing.ego_motion import EgoMotionEstimator

CONFIG_PATH  = Path(__file__).parent.parent / 'config' / 'default.yaml'
KITTI_ROOT   = Path('~/kitti_dataset').expanduser() / 'training'
EVAL_CLASSES = {'Car', 'Van', 'Pedestrian', 'Cyclist'}


@dataclass
class _SensorTrack:
    track_id: int
    last_detection: Detection


# ── KITTI coordinate helpers ──────────────────────────────────────────────────

def cam_to_velo(points_cam, calib):
    R_full = calib.rect @ calib.velo_to_cam[:, :3]
    t_full = calib.rect @ calib.velo_to_cam[:, 3]
    return (R_full.T @ points_cam.T).T + (-R_full.T @ t_full)


def gt_to_lidar(det, calib):
    center_velo = cam_to_velo(np.array([[det.x, det.y - det.height / 2.0, det.z]]), calib)[0]
    return KittiDetection(
        track_id=det.track_id, object_type=det.object_type,
        height=det.height, width=det.width, length=det.length,
        x=float(center_velo[0]), y=float(center_velo[1]),
        z=float(center_velo[2] - det.height / 2.0),
        rotation_y=-det.rotation_y,
    )


# ── Per-sequence data preparation ─────────────────────────────────────────────

def prepare_sequence(seq: str, config: dict, detector_mode: str, conf: float | None):
    """
    Returns:
        gt_lidar          – {frame_id: [KittiDetection]}  in sensor frame
        world_dets        – [list[Detection]] per frame, in world frame
        ego_states        – [(R_ws, t_ws)] per frame
    """
    calib    = load_calibration(KITTI_ROOT / 'calib'    / f'{seq}.txt')
    labels   = load_labels    (KITTI_ROOT / 'label_02' / f'{seq}.txt')
    gt_lidar = {
        fid: [gt_to_lidar(d, calib) for d in dets if d.object_type in EVAL_CLASSES]
        for fid, dets in labels.items()
        if any(d.object_type in EVAL_CLASSES for d in dets)
    }

    frame_files = sorted((KITTI_ROOT / 'velodyne' / seq).glob('*.bin'))
    pre_cfg = config['preprocessing']

    # ── Detect ────────────────────────────────────────────────────────────────
    if detector_mode == 'point_pillars':
        from lidar_tracker.core.detection.point_pillars import PointPillarsDetector
        pp = config['detection']['point_pillars']
        detector = PointPillarsDetector(
            Path(pp['model_path']),
            conf_threshold=conf,
            nms_threshold=pp['nms_threshold'],
            device=pp['device'],
        )
        do_preprocess = False
    else:
        from lidar_tracker.core.detection.euclidean import EuclideanDetector
        ec = config['detection']['euclidean']
        detector = EuclideanDetector(
            eps=ec['eps'], min_points=ec['min_points'],
            min_h=ec['min_h'], max_h=ec['max_h'],
            min_w=ec['min_w'], max_w=ec['max_w'],
            min_l=ec['min_l'], max_l=ec['max_l'],
            max_center_z=ec['max_center_z'],
        )
        do_preprocess = True

    ego = EgoMotionEstimator()
    ego_states  = []
    world_dets  = []

    for f in frame_files:
        pts = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
        ego.update(pts)
        ego_states.append((ego.R_ws.copy(), ego.t_ws.copy()))

        if do_preprocess:
            pts = range_crop(pts, pre_cfg['min_distance'], pre_cfg['max_distance'])
            pts = remove_ground(pts, pre_cfg['ground_threshold'])
            pts = voxel_downsample(pts, pre_cfg['voxel_size'])

        sensor_dets = detector.detect(pts)
        R_ws, t_ws  = ego_states[-1]
        frame_world = []
        for d in sensor_dets:
            pos_w = R_ws @ d.position + t_ws
            frame_world.append(Detection(
                x=float(pos_w[0]), y=float(pos_w[1]), z=float(pos_w[2]),
                width=d.width, length=d.length, height=d.height,
                rotation_y=d.rotation_y, confidence=d.confidence,
                object_type=d.object_type,
            ))
        world_dets.append(frame_world)

    return gt_lidar, world_dets, ego_states


def run_tracker_multi(seq_data, min_hits, max_age, match_dist, eval_match_dist):
    """Average metrics across multiple sequences."""
    agg = {}
    for gt_lidar, world_dets, ego_states in seq_data:
        trk = Sort3D(max_age=max_age, min_hits=min_hits, match_distance=match_dist)
        all_tracks = {}
        for fi, (wdets, (R_ws, t_ws)) in enumerate(zip(world_dets, ego_states)):
            tracks_world = trk.update(wdets, dt=0.1)
            sensor_tracks = []
            for t in tracks_world:
                pos_s = R_ws.T @ (t.state[:3] - t_ws)
                ld    = t.last_detection
                sensor_tracks.append(_SensorTrack(
                    track_id=t.track_id,
                    last_detection=Detection(
                        x=float(pos_s[0]), y=float(pos_s[1]), z=float(pos_s[2]),
                        width=ld.width, length=ld.length, height=ld.height,
                        rotation_y=ld.rotation_y, confidence=ld.confidence,
                        object_type=ld.object_type,
                    ),
                ))
            all_tracks[fi] = sensor_tracks
        m = compute_metrics(all_gt=gt_lidar, all_tracks=all_tracks, match_distance=eval_match_dist)
        for k, v in m.items():
            agg[k] = agg.get(k, 0) + v
    n = len(seq_data)
    return {k: v / n for k, v in agg.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequences', nargs='+', default=['0000'],
                        help='One or more KITTI sequence IDs to evaluate over')
    parser.add_argument('--detector', choices=['point_pillars', 'euclidean'],
                        default='point_pillars')
    parser.add_argument('--write-config', action='store_true',
                        help='Write best params back to config/default.yaml')
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text())
    eval_match_dist = config['evaluation']['match_distance']

    # ── Search grids ──────────────────────────────────────────────────────────
    min_hits_vals   = [1, 2, 3]
    max_age_vals    = [2, 3, 5, 7, 10]
    match_dist_vals = [0.5, 1.0, 1.5, 2.0, 3.0]
    conf_vals       = [0.3, 0.4, 0.5, 0.6, 0.7] if args.detector == 'point_pillars' else [None]

    total_combos = len(conf_vals) * len(min_hits_vals) * len(max_age_vals) * len(match_dist_vals)
    print(f'Detector : {args.detector}')
    print(f'Sequences: {args.sequences}')
    print(f'Grid     : {total_combos} combinations\n')

    results = []
    best_mota = -float('inf')
    best_cfg  = {}

    for conf in conf_vals:
        label = f'conf={conf:.1f}' if conf is not None else 'euclidean'
        print(f'── Detecting ({label}) across {len(args.sequences)} sequence(s)...')
        seq_data = []
        for seq in args.sequences:
            gt, wdets, estates = prepare_sequence(seq, config, args.detector, conf)
            seq_data.append((gt, wdets, estates))
        print(f'   Done. Running {len(min_hits_vals)*len(max_age_vals)*len(match_dist_vals)} tracker combos...')

        for min_hits, max_age, match_dist in product(min_hits_vals, max_age_vals, match_dist_vals):
            m = run_tracker_multi(seq_data, min_hits, max_age, match_dist, eval_match_dist)
            mota = m['MOTA']
            cfg  = dict(conf=conf, min_hits=min_hits, max_age=max_age, match_dist=match_dist)
            results.append((mota, m, cfg))
            if mota > best_mota:
                best_mota = mota
                best_cfg  = cfg

    # ── Print top 15 ─────────────────────────────────────────────────────────
    results.sort(key=lambda x: x[0], reverse=True)
    print(f'\n{"─"*85}')
    print(f'{"Rank":>4}  {"MOTA":>7}  {"MOTP":>6}  {"IDS":>5}  {"FP":>6}  {"FN":>6}  '
          f'{"conf":>5}  {"hits":>4}  {"age":>4}  {"mdist":>6}')
    print(f'{"─"*85}')
    for rank, (mota, m, cfg) in enumerate(results[:15], 1):
        conf_str = f'{cfg["conf"]:.1f}' if cfg['conf'] is not None else '  n/a'
        print(f'{rank:>4}  {mota:>+7.4f}  {m["MOTP (mean center dist m)"]:>6.3f}  '
              f'{m["ID Switches"]:>5.0f}  {m["False Positives"]:>6.0f}  {m["False Negatives"]:>6.0f}  '
              f'{conf_str:>5}  {cfg["min_hits"]:>4}  {cfg["max_age"]:>4}  {cfg["match_dist"]:>6.1f}')
    print(f'{"─"*85}')

    print(f'\nBest MOTA : {best_mota:+.4f}')
    print(f'Best config: {best_cfg}')

    if args.write_config:
        mode = args.detector
        config['tracking'][mode]['min_hits']      = best_cfg['min_hits']
        config['tracking'][mode]['max_age']        = best_cfg['max_age']
        config['tracking'][mode]['match_distance'] = best_cfg['match_dist']
        if best_cfg['conf'] is not None:
            config['detection'][mode]['conf_threshold'] = best_cfg['conf']
        CONFIG_PATH.write_text(yaml.dump(config, default_flow_style=False))
        print(f'\nWrote best params to {CONFIG_PATH}')


if __name__ == '__main__':
    main()
