#!/usr/bin/env python3
"""
Compare all detector modes across all KITTI training sequences.

When point_pillars and pvrcnn are both requested, they are loaded once and
their per-frame detections are cached.  Fusion NMS is swept over multiple
distances at no extra model cost; standalone PP and PV-RCNN metrics come from
the same cached detections.

Usage:
    PYTHONPATH=. python scripts/benchmark.py
    PYTHONPATH=. python scripts/benchmark.py --detectors point_pillars pvrcnn
    PYTHONPATH=. python scripts/benchmark.py --nms 0.5 1.0 1.5 2.0 2.5
"""
import argparse
import sys
import numpy as np
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_tracker.core.data.kitti_loader import load_calibration, load_labels, KittiDetection
from lidar_tracker.core.detection.base import Detection
from lidar_tracker.core.detection.fusion import _nms
from lidar_tracker.core.tracking.sort3d import Sort3D
from lidar_tracker.core.evaluation.metrics import (
    compute_metrics, compute_hota, compute_mt_pt_ml, compute_idf1,
)
from lidar_tracker.core.preprocessing.filters import range_crop, voxel_downsample
from lidar_tracker.core.preprocessing.ground_filter import remove_ground
from lidar_tracker.core.preprocessing.ego_motion import EgoMotionEstimator

CONFIG_PATH  = Path(__file__).parent.parent / 'config' / 'default.yaml'
KITTI_ROOT   = Path('~/kitti_dataset').expanduser() / 'training'
EVAL_CLASSES = {'Car', 'Pedestrian', 'Cyclist'}
DEFAULT_NMS  = [0.5, 1.0, 1.5, 2.0, 2.5]


@dataclass
class _SensorTrack:
    track_id: int
    last_detection: Detection


# ── KITTI helpers ──────────────────────────────────────────────────────────────

def _cam_to_velo(pts_cam, calib):
    R = calib.rect @ calib.velo_to_cam[:, :3]
    t = calib.rect @ calib.velo_to_cam[:, 3]
    return (R.T @ pts_cam.T).T + (-R.T @ t)


def _gt_to_lidar(det, calib):
    c = _cam_to_velo(np.array([[det.x, det.y - det.height / 2.0, det.z]]), calib)[0]
    return KittiDetection(
        track_id=det.track_id, object_type=det.object_type,
        height=det.height, width=det.width, length=det.length,
        x=float(c[0]), y=float(c[1]), z=float(c[2] - det.height / 2.0),
        rotation_y=-det.rotation_y,
    )


def _discover_sequences():
    seqs = []
    for d in sorted((KITTI_ROOT / 'velodyne').iterdir()):
        if d.is_dir() and (KITTI_ROOT / 'label_02' / f'{d.name}.txt').exists():
            seqs.append(d.name)
    return seqs


# ── Detector loading ───────────────────────────────────────────────────────────

def _load_detector(mode, config):
    if mode == 'euclidean':
        from lidar_tracker.core.detection.euclidean import EuclideanDetector
        ec = config['detection']['euclidean']
        return EuclideanDetector(
            eps=ec['eps'], min_points=ec['min_points'],
            min_h=ec['min_h'], max_h=ec['max_h'],
            min_w=ec['min_w'], max_w=ec['max_w'],
            min_l=ec['min_l'], max_l=ec['max_l'],
            max_center_z=ec['max_center_z'],
        )
    elif mode == 'point_pillars':
        from lidar_tracker.core.detection.point_pillars import PointPillarsDetector
        pp = config['detection']['point_pillars']
        return PointPillarsDetector(
            Path(pp['model_path']),
            conf_threshold=pp['conf_threshold'],
            nms_threshold=pp['nms_threshold'],
            device=pp['device'],
        )
    elif mode in ('pvrcnn', 'second', 'voxel_rcnn'):
        from lidar_tracker.core.detection.openpcdet_detector import OpenPCDetDetector
        dc = config['detection'][mode]
        return OpenPCDetDetector(
            cfg_file=Path(dc['cfg_file']),
            ckpt_path=Path(dc['model_path']),
            conf_threshold=dc['conf_threshold'],
            device=dc['device'],
        )
    raise ValueError(f'Unknown mode: {mode}')


# ── Core building blocks ───────────────────────────────────────────────────────

def _to_world(dets, R_ws, t_ws):
    out = []
    for d in dets:
        pos_w = R_ws @ d.position + t_ws
        out.append(Detection(
            x=float(pos_w[0]), y=float(pos_w[1]), z=float(pos_w[2]),
            width=d.width, length=d.length, height=d.height,
            rotation_y=d.rotation_y, confidence=d.confidence,
            object_type=d.object_type,
        ))
    return out


def _track_and_eval(gt_lidar, world_dets_by_frame, ego_cache, trk_cfg, eval_md):
    """Run Sort3D on cached world-frame detections; convert tracks back to sensor frame for eval."""
    tracker = Sort3D(
        max_age=trk_cfg['max_age'],
        min_hits=trk_cfg['min_hits'],
        match_distance=trk_cfg['match_distance'],
    )
    all_tracks: dict[int, list] = {}
    for fi in sorted(world_dets_by_frame):
        R_ws, t_ws = ego_cache[fi]
        sensor_tracks = []
        for t in tracker.update(world_dets_by_frame[fi], dt=0.1):
            pos_s = R_ws.T @ (t.state[:3] - t_ws)
            ld = t.last_detection
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
    h   = compute_hota(gt_lidar, all_tracks)
    m   = compute_metrics(gt_lidar, all_tracks, match_distance=eval_md)
    mt  = compute_mt_pt_ml(gt_lidar, all_tracks, match_distance=eval_md)
    idf = compute_idf1(gt_lidar, all_tracks, match_distance=eval_md)
    return {**h, **m, **mt, **idf}


# ── Joint PP + PV-RCNN evaluation (shared detection pass) ────────────────────

def _detect_sequence(seq, pp_det, pv_det, config):
    """Run PP and PV-RCNN once per frame, return GT + cached world-frame detections."""
    calib  = load_calibration(KITTI_ROOT / 'calib'    / f'{seq}.txt')
    labels = load_labels    (KITTI_ROOT / 'label_02' / f'{seq}.txt')
    gt_lidar = {
        fid: [_gt_to_lidar(d, calib) for d in dets if d.object_type in EVAL_CLASSES]
        for fid, dets in labels.items()
        if any(d.object_type in EVAL_CLASSES for d in dets)
    }

    frame_files = sorted((KITTI_ROOT / 'velodyne' / seq).glob('*.bin'))
    ego = EgoMotionEstimator()

    ego_cache:  dict[int, tuple] = {}
    pp_world:   dict[int, list]  = {}
    pv_world:   dict[int, list]  = {}

    for fi, f in enumerate(frame_files):
        pts = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
        ego.update(pts)
        R_ws, t_ws = ego.R_ws.copy(), ego.t_ws.copy()
        ego_cache[fi] = (R_ws, t_ws)

        pp_world[fi] = _to_world(pp_det.detect(pts), R_ws, t_ws)
        pv_world[fi] = _to_world(pv_det.detect(pts), R_ws, t_ws)

    return gt_lidar, ego_cache, pp_world, pv_world


def benchmark_joint(config, sequences, nms_vals):
    """
    Load PP and PV-RCNN once, run both on every frame, then evaluate:
      - PP standalone
      - PV-RCNN standalone
      - Fusion at each NMS distance in nms_vals
    Returns {mode_label: avg_metrics}.
    """
    print('\n  Loading point_pillars...')
    try:
        pp_det = _load_detector('point_pillars', config)
    except Exception as e:
        print(f'  SKIP point_pillars (load failed): {e}')
        return {}

    print('  Loading pvrcnn...')
    try:
        pv_det = _load_detector('pvrcnn', config)
    except Exception as e:
        print(f'  SKIP pvrcnn (load failed): {e}')
        return {}

    eval_md  = config['evaluation']['match_distance']
    pp_trk   = config['tracking']['point_pillars']
    pv_trk   = config['tracking']['pvrcnn']
    fu_trk   = config['tracking']['fusion']
    fusion_keys = [f'fusion_{d:.1f}m' for d in nms_vals]

    totals: dict[str, dict[str, float]] = {}
    n = 0

    for seq in sequences:
        try:
            gt_lidar, ego_cache, pp_world, pv_world = _detect_sequence(
                seq, pp_det, pv_det, config)

            r_pp = _track_and_eval(gt_lidar, pp_world, ego_cache, pp_trk, eval_md)
            r_pv = _track_and_eval(gt_lidar, pv_world, ego_cache, pv_trk, eval_md)
            seq_results = {'point_pillars': r_pp, 'pvrcnn': r_pv}

            all_frames = set(pp_world) | set(pv_world)
            for nms_d, key in zip(nms_vals, fusion_keys):
                fused = {
                    fi: _nms(pp_world.get(fi, []) + pv_world.get(fi, []), nms_d)
                    for fi in all_frames
                }
                seq_results[key] = _track_and_eval(gt_lidar, fused, ego_cache, fu_trk, eval_md)

            for mode, r in seq_results.items():
                totals.setdefault(mode, {})
                for k, v in r.items():
                    totals[mode][k] = totals[mode].get(k, 0.0) + v
            n += 1

            fusion_str = '  '.join(
                f'F@{d:.1f}={seq_results[key]["HOTA"]:.3f}'
                for d, key in zip(nms_vals, fusion_keys)
            )
            print(f'    {seq}  PP={r_pp["HOTA"]:.3f}  PV={r_pv["HOTA"]:.3f}  {fusion_str}')

        except Exception as e:
            print(f'    {seq}  SKIP: {e}')

    if n == 0:
        return {}
    avgs = {mode: {k: v / n for k, v in m.items()} for mode, m in totals.items()}
    for mode in avgs:
        avgs[mode]['n_sequences'] = n
    return avgs


# ── Standalone benchmark (euclidean / other single-model modes) ───────────────

def _eval_sequence_standalone(seq, detector, mode, config):
    calib  = load_calibration(KITTI_ROOT / 'calib'    / f'{seq}.txt')
    labels = load_labels    (KITTI_ROOT / 'label_02' / f'{seq}.txt')
    gt_lidar = {
        fid: [_gt_to_lidar(d, calib) for d in dets if d.object_type in EVAL_CLASSES]
        for fid, dets in labels.items()
        if any(d.object_type in EVAL_CLASSES for d in dets)
    }
    frame_files = sorted((KITTI_ROOT / 'velodyne' / seq).glob('*.bin'))
    pre_cfg     = config['preprocessing']
    trk_cfg     = config['tracking'][mode]
    do_preprocess = (mode == 'euclidean')
    ego = EgoMotionEstimator()
    ego_cache:   dict[int, tuple] = {}
    world_dets_by_frame: dict[int, list] = {}

    for fi, f in enumerate(frame_files):
        pts = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
        ego.update(pts)
        R_ws, t_ws = ego.R_ws.copy(), ego.t_ws.copy()
        ego_cache[fi] = (R_ws, t_ws)
        det_pts = pts
        if do_preprocess:
            det_pts = range_crop(det_pts, pre_cfg['min_distance'], pre_cfg['max_distance'])
            det_pts = remove_ground(det_pts, pre_cfg['ground_threshold'])
            det_pts = voxel_downsample(det_pts, pre_cfg['voxel_size'])
        world_dets_by_frame[fi] = _to_world(detector.detect(det_pts), R_ws, t_ws)

    return _track_and_eval(gt_lidar, world_dets_by_frame, ego_cache, trk_cfg, eval_md=config['evaluation']['match_distance'])


def benchmark_standalone(mode, config, sequences):
    print(f'\n  Loading {mode}...')
    try:
        detector = _load_detector(mode, config)
    except Exception as e:
        print(f'  SKIP (load failed): {e}')
        return None

    totals: dict[str, float] = {}
    n = 0
    for seq in sequences:
        try:
            r = _eval_sequence_standalone(seq, detector, mode, config)
            for k, v in r.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1
            print(f'    {seq}  HOTA={r["HOTA"]:.3f}  '
                  f'FP={int(r["False Positives"])}  FN={int(r["False Negatives"])}')
        except Exception as e:
            print(f'    {seq}  SKIP: {e}')

    if n == 0:
        return None
    avg = {k: v / n for k, v in totals.items()}
    avg['n_sequences'] = n
    return avg


# ── Results table ──────────────────────────────────────────────────────────────

def _print_table(results: dict[str, dict], nms_vals: list[float]):
    fusion_keys = {f'fusion_{d:.1f}m' for d in nms_vals}
    best_fusion = max(
        (k for k in results if k in fusion_keys),
        key=lambda k: results[k].get('HOTA', 0.0),
        default=None,
    )

    W = 112
    print(f'\n\n{"═"*W}')
    print(f'{"BENCHMARK — averaged across sequences":^{W}}')
    print(f'{"═"*W}')
    hdr = (f'{"Detector":<18}  {"HOTA":>7}  {"DetA":>7}  {"AssA":>7}  {"IDF1":>7}  '
           f'{"MOTA":>7}  {"MT%":>6}  {"ML%":>6}  {"IDS":>5}  {"FP":>7}  {"FN":>7}  {"N":>3}')
    print(hdr)
    print(f'{"─"*W}')

    def _row(label, r, marker=''):
        return (f'{label+marker:<18}  {r["HOTA"]:>7.4f}  {r["DetA"]:>7.4f}  {r["AssA"]:>7.4f}'
                f'  {r["IDF1"]:>7.4f}  {r["MOTA"]:>+7.4f}'
                f'  {r["MT%"]*100:>5.1f}%  {r["ML%"]*100:>5.1f}%'
                f'  {int(r["ID Switches"]):>5}  {int(r["False Positives"]):>7}'
                f'  {int(r["False Negatives"]):>7}  {int(r["n_sequences"]):>3}')

    # Non-fusion rows first
    for mode, r in results.items():
        if mode not in fusion_keys:
            print(_row(mode, r))

    # Fusion rows, best marked with *
    if any(k in fusion_keys for k in results):
        print(f'{"─"*W}')
        for nms_d in nms_vals:
            key = f'fusion_{nms_d:.1f}m'
            if key in results:
                marker = ' *' if key == best_fusion else ''
                print(_row(key, results[key], marker))
        if best_fusion:
            print(f'\n  * best fusion NMS distance by HOTA')

    print(f'{"═"*W}')


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Multi-detector KITTI benchmark')
    parser.add_argument('--detectors', nargs='+',
                        default=['euclidean', 'point_pillars', 'pvrcnn'],
                        choices=['euclidean', 'point_pillars', 'pvrcnn',
                                 'second', 'voxel_rcnn'])
    parser.add_argument('--nms', nargs='+', type=float, default=DEFAULT_NMS,
                        metavar='DIST',
                        help='NMS distances (m) to sweep for fusion (default: 0.5 1.0 1.5 2.0 2.5)')
    parser.add_argument('--no-fusion', action='store_true',
                        help='Skip fusion; benchmark each detector independently')
    args = parser.parse_args()

    config    = yaml.safe_load(CONFIG_PATH.read_text())
    sequences = _discover_sequences()
    print(f'Found {len(sequences)} sequences: {sequences}')

    results: dict[str, dict] = {}
    run_joint = (not args.no_fusion
                 and 'point_pillars' in args.detectors
                 and 'pvrcnn' in args.detectors)

    # Standalone detectors (euclidean, or when --no-fusion)
    standalone = [m for m in args.detectors if m not in ('point_pillars', 'pvrcnn')] \
        if run_joint else args.detectors
    for mode in standalone:
        print(f'\n{"─"*60}\nDetector: {mode}')
        r = benchmark_standalone(mode, config, sequences)
        if r:
            results[mode] = r

    # Joint PP + PV-RCNN + fusion sweep
    if run_joint:
        print(f'\n{"─"*60}')
        print(f'Joint evaluation: point_pillars  pvrcnn  fusion @ NMS {args.nms}')
        joint = benchmark_joint(config, sequences, args.nms)
        results.update(joint)

    if not results:
        print('\nNo results to report.')
        return

    _print_table(results, args.nms)


if __name__ == '__main__':
    main()
