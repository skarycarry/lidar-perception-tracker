#!/usr/bin/env python3
"""
Optimise tracker hyperparameters using Bayesian optimisation (Optuna).

Detection is expensive, so world-frame detections are cached per
conf-threshold × sequence up front.  Optuna then explores the tracker
parameter space (min_hits, max_age, match_distance, conf) cheaply.

Falls back to exhaustive grid search if Optuna is not installed.

Usage:
    pip install optuna          # one-time
    python scripts/tune_tracker.py
    python scripts/tune_tracker.py --detector euclidean
    python scripts/tune_tracker.py --trials 200
    python scripts/tune_tracker.py --write-config   # save best to default.yaml
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
from lidar_tracker.core.evaluation.metrics import (
    compute_metrics, compute_hota, compute_mt_pt_ml, compute_idf1,
)
from lidar_tracker.core.preprocessing.filters import range_crop, voxel_downsample
from lidar_tracker.core.preprocessing.ground_filter import remove_ground
from lidar_tracker.core.preprocessing.ego_motion import EgoMotionEstimator

CONFIG_PATH  = Path(__file__).parent.parent / 'config' / 'default.yaml'
KITTI_ROOT   = Path('~/kitti_dataset').expanduser() / 'training'
EVAL_CLASSES = {'Car', 'Pedestrian', 'Cyclist'}  # Van excluded: no PointPillars class support

# Conf thresholds pre-cached for Bayesian search
CONF_VALS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class _SensorTrack:
    track_id: int
    last_detection: Detection


# ── KITTI helpers ─────────────────────────────────────────────────────────────

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


def discover_sequences() -> list[str]:
    """Return all sequence IDs that have both velodyne frames and labels."""
    velodyne_dir = KITTI_ROOT / 'velodyne'
    label_dir    = KITTI_ROOT / 'label_02'
    seqs = []
    for d in sorted(velodyne_dir.iterdir()):
        if d.is_dir() and (label_dir / f'{d.name}.txt').exists():
            seqs.append(d.name)
    return seqs


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_sequence(seq: str, config: dict, detector_mode: str, conf: float | None):
    """
    Returns:
        gt_lidar   – {frame_id: [KittiDetection]} in sensor frame
        world_dets – [list[Detection]] per frame in world frame
        ego_states – [(R_ws, t_ws)] per frame
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

    if detector_mode == 'point_pillars':
        from lidar_tracker.core.detection.point_pillars import PointPillarsDetector
        pp = config['detection']['point_pillars']
        detector = PointPillarsDetector(
            Path(pp['model_path']), conf_threshold=conf,
            nms_threshold=pp['nms_threshold'], device=pp['device'],
        )
        do_preprocess = False
    elif detector_mode in ('second', 'pvrcnn', 'voxel_rcnn'):
        from lidar_tracker.core.detection.openpcdet_detector import OpenPCDetDetector
        dc = config['detection'][detector_mode]
        detector = OpenPCDetDetector(
            cfg_file=Path(dc['cfg_file']),
            ckpt_path=Path(dc['model_path']),
            conf_threshold=conf,
            device=dc['device'],
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

    ego        = EgoMotionEstimator()
    ego_states = []
    world_dets = []

    for f in frame_files:
        pts = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
        ego.update(pts)
        ego_states.append((ego.R_ws.copy(), ego.t_ws.copy()))

        det_pts = pts
        if do_preprocess:
            det_pts = range_crop(det_pts, pre_cfg['min_distance'], pre_cfg['max_distance'])
            det_pts = remove_ground(det_pts, pre_cfg['ground_threshold'])
            det_pts = voxel_downsample(det_pts, pre_cfg['voxel_size'])

        R_ws, t_ws = ego_states[-1]
        frame_world = []
        for d in detector.detect(det_pts):
            pos_w = R_ws @ d.position + t_ws
            frame_world.append(Detection(
                x=float(pos_w[0]), y=float(pos_w[1]), z=float(pos_w[2]),
                width=d.width, length=d.length, height=d.height,
                rotation_y=d.rotation_y, confidence=d.confidence,
                object_type=d.object_type,
            ))
        world_dets.append(frame_world)

    return gt_lidar, world_dets, ego_states


# ── Tracker evaluation ────────────────────────────────────────────────────────

def evaluate(seq_data, min_hits: int, max_age: int, match_dist: float,
             eval_match_dist: float) -> dict:
    """Run tracker over all sequences, return HOTA + MOTA + IDF1 + MT/ML averaged across sequences."""
    hota_sum = deta_sum = assa_sum = idf1_sum = mt_pct_sum = ml_pct_sum = 0.0
    mota_totals: dict = {}

    for gt_lidar, world_dets, ego_states in seq_data:
        trk = Sort3D(max_age=max_age, min_hits=min_hits, match_distance=match_dist)
        all_tracks: dict = {}
        for fi, (wdets, (R_ws, t_ws)) in enumerate(zip(world_dets, ego_states)):
            sensor_tracks = []
            for t in trk.update(wdets, dt=0.1):
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

        h = compute_hota(gt_lidar, all_tracks)
        hota_sum += h['HOTA']
        deta_sum += h['DetA']
        assa_sum += h['AssA']

        m = compute_metrics(gt_lidar, all_tracks, match_distance=eval_match_dist)
        for k, v in m.items():
            mota_totals[k] = mota_totals.get(k, 0.0) + v

        idf = compute_idf1(gt_lidar, all_tracks, match_distance=eval_match_dist)
        idf1_sum += idf['IDF1']

        mt = compute_mt_pt_ml(gt_lidar, all_tracks, match_distance=eval_match_dist)
        mt_pct_sum += mt['MT%']
        ml_pct_sum += mt['ML%']

    n = len(seq_data)
    result = {k: v / n for k, v in mota_totals.items()}
    result.update({
        'HOTA': hota_sum / n,
        'DetA': deta_sum / n,
        'AssA': assa_sum / n,
        'IDF1': idf1_sum / n,
        'MT%':  mt_pct_sum / n,
        'ML%':  ml_pct_sum / n,
    })
    return result


# ── Search strategies ─────────────────────────────────────────────────────────

def run_bayesian(cache, conf_vals, eval_match_dist, n_trials, timeout_secs, study_name):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    db_path  = Path(__file__).parent.parent / 'tune_results.db'
    storage  = f'sqlite:///{db_path}'
    study    = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        load_if_exists=True,   # resume automatically if study already exists
    )

    existing = len(study.trials)
    if existing:
        print(f'  Resuming study "{study_name}" — {existing} trials already completed.')

    results = []

    def objective(trial):
        conf_key   = trial.suggest_categorical('conf', conf_vals)
        min_hits   = trial.suggest_int  ('min_hits',    1,   8)
        max_age    = trial.suggest_int  ('max_age',     1,  30)
        match_dist = trial.suggest_float('match_dist', 0.1,  8.0)

        m    = evaluate(cache[conf_key], min_hits, max_age, match_dist, eval_match_dist)
        cfg  = dict(conf=conf_key, min_hits=min_hits, max_age=max_age, match_dist=match_dist)
        results.append((m['HOTA'], m, cfg))
        return m['HOTA']

    study.optimize(objective, n_trials=n_trials, timeout=timeout_secs,
                   show_progress_bar=True)
    print(f'\n  Results saved to {db_path}')

    # Reconstruct results list from all trials (including any from previous runs)
    all_results = []
    for t in study.trials:
        if t.value is None:
            continue
        p = t.params
        cfg = dict(conf=p['conf'], min_hits=p['min_hits'],
                   max_age=p['max_age'], match_dist=p['match_dist'])
        # Re-evaluate to get full metrics dict for the best trial only
        all_results.append((t.value, cfg))

    best = study.best_params
    return results, dict(
        conf       = best['conf'],
        min_hits   = best['min_hits'],
        max_age    = best['max_age'],
        match_dist = best['match_dist'],
    )


def run_grid(cache, conf_vals, eval_match_dist):
    min_hits_vals   = [1, 2, 3, 5, 8]
    max_age_vals    = [1, 2, 3, 5, 8, 12, 20, 30]
    match_dist_vals = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]

    results   = []
    best_mota = -float('inf')
    best_cfg  = {}

    total = len(conf_vals) * len(min_hits_vals) * len(max_age_vals) * len(match_dist_vals)
    done  = 0
    for conf, min_hits, max_age, match_dist in product(
            conf_vals, min_hits_vals, max_age_vals, match_dist_vals):
        m     = evaluate(cache[conf], min_hits, max_age, match_dist, eval_match_dist)
        cfg   = dict(conf=conf, min_hits=min_hits, max_age=max_age, match_dist=match_dist)
        score = m['HOTA']
        results.append((score, m, cfg))
        if score > best_mota:
            best_mota = score
            best_cfg  = cfg
        done += 1
        print(f'\r  {done}/{total}  best MOTA so far: {best_mota:+.4f}', end='', flush=True)
    print()
    return results, best_cfg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector',
                        choices=['point_pillars', 'euclidean', 'pvrcnn', 'second', 'voxel_rcnn'],
                        default='point_pillars')
    parser.add_argument('--trials', type=int, default=1000,
                        help='Max Bayesian optimisation trials')
    parser.add_argument('--hours', type=float, default=None,
                        help='Stop after this many hours (whichever comes first: --trials or --hours)')
    parser.add_argument('--study-name', default='lpt_tracker',
                        help='Optuna study name — reuse to resume a previous run')
    parser.add_argument('--write-config', action='store_true',
                        help='Write best params back to config/default.yaml')
    args = parser.parse_args()

    config          = yaml.safe_load(CONFIG_PATH.read_text())
    eval_match_dist = config['evaluation']['match_distance']
    sequences       = discover_sequences()
    _uses_conf = ('point_pillars', 'pvrcnn', 'second', 'voxel_rcnn')
    conf_vals  = CONF_VALS if args.detector in _uses_conf else [None]

    print(f'Detector  : {args.detector}')
    print(f'Sequences : {sequences}  ({len(sequences)} total)')
    print(f'Conf vals : {conf_vals}')

    # ── Cache detections ──────────────────────────────────────────────────────
    print(f'\nPre-caching detections ({len(conf_vals)} conf × {len(sequences)} sequences)...')
    cache: dict[float | None, list] = {}
    for conf in conf_vals:
        label = f'conf={conf}' if conf is not None else 'euclidean'
        print(f'  {label}')
        cache[conf] = []
        for seq in sequences:
            try:
                cache[conf].append(prepare_sequence(seq, config, args.detector, conf))
            except Exception as e:
                print(f'    skipping {seq}: {e}')

    # ── Optimise ──────────────────────────────────────────────────────────────
    try:
        import optuna  # noqa: F401
        timeout_secs = int(args.hours * 3600) if args.hours else None
        limit_str    = (f'{args.trials} trials' +
                        (f' or {args.hours}h' if args.hours else ''))
        print(f'\nRunning Bayesian optimisation ({limit_str})...')
        results, best_cfg = run_bayesian(cache, conf_vals, eval_match_dist,
                                         args.trials, timeout_secs, args.study_name)
    except ImportError:
        print('\nOptuna not found — falling back to grid search.')
        print('Install with:  pip install optuna\n')
        results, best_cfg = run_grid(cache, conf_vals, eval_match_dist)

    # ── Report ────────────────────────────────────────────────────────────────
    results.sort(key=lambda x: x[0], reverse=True)
    W = 120
    print(f'\n{"─"*W}')
    print(f'{"Rank":>4}  {"HOTA":>7}  {"DetA":>7}  {"AssA":>7}  {"IDF1":>7}  {"MOTA":>7}  '
          f'{"MT%":>6}  {"ML%":>6}  {"IDS":>5}  {"FP":>6}  {"FN":>6}  '
          f'{"conf":>5}  {"hits":>4}  {"age":>4}  {"mdist":>6}')
    print(f'{"─"*W}')
    for rank, (score, m, cfg) in enumerate(results[:15], 1):
        conf_str = f'{cfg["conf"]:.1f}' if cfg['conf'] is not None else '  n/a'
        print(f'{rank:>4}  {m["HOTA"]:>7.4f}  {m["DetA"]:>7.4f}  {m["AssA"]:>7.4f}'
              f'  {m["IDF1"]:>7.4f}  {m["MOTA"]:>+7.4f}'
              f'  {m["MT%"]*100:5.1f}%  {m["ML%"]*100:5.1f}%'
              f'  {int(m["ID Switches"]):>5}  {int(m["False Positives"]):>6}'
              f'  {int(m["False Negatives"]):>6}'
              f'  {conf_str:>5}  {cfg["min_hits"]:>4}  {cfg["max_age"]:>4}'
              f'  {cfg["match_dist"]:>6.2f}')
    print(f'{"─"*W}')
    print(f'\nBest HOTA : {results[0][0]:.4f}')
    print(f'Best config: {best_cfg}')

    if args.write_config:
        mode = args.detector
        config['tracking'][mode]['min_hits']      = int(best_cfg['min_hits'])
        config['tracking'][mode]['max_age']        = int(best_cfg['max_age'])
        config['tracking'][mode]['match_distance'] = round(float(best_cfg['match_dist']), 3)
        if best_cfg['conf'] is not None:
            config['detection'][mode]['conf_threshold'] = float(best_cfg['conf'])
        CONFIG_PATH.write_text(yaml.dump(config, default_flow_style=False))
        print(f'\nWrote best params to {CONFIG_PATH}')


if __name__ == '__main__':
    main()
