"""Unit tests for evaluation metrics (MOTA, HOTA, MT/ML, IDF1, per-class)."""
import numpy as np
import pytest
from dataclasses import dataclass

from lidar_tracker.core.data.kitti_loader import KittiDetection
from lidar_tracker.core.detection.base import Detection
from lidar_tracker.core.evaluation.metrics import (
    compute_metrics,
    compute_hota,
    compute_mt_pt_ml,
    compute_idf1,
    compute_per_class,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gt(track_id, x, y, z=0.0, h=1.0, w=1.0, l=2.0, cls='Car'):
    return KittiDetection(
        track_id=track_id, object_type=cls,
        x=x, y=y, z=z, height=h, width=w, length=l, rotation_y=0.0,
    )


@dataclass
class _Track:
    track_id: int
    last_detection: Detection


def _trk(track_id, x, y, z=0.0, h=1.0, w=1.0, l=2.0, cls='Car'):
    det = Detection(x=x, y=y, z=z, height=h, width=w, length=l,
                    rotation_y=0.0, confidence=1.0, object_type=cls)
    return _Track(track_id=track_id, last_detection=det)


MATCH_D = 2.0


# ── compute_metrics (MOTA) ────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_perfect_tracking(self):
        gt = {0: [_gt(1, 0, 0)], 1: [_gt(1, 1, 0)]}
        tr = {0: [_trk(10, 0, 0)], 1: [_trk(10, 1, 0)]}
        m = compute_metrics(gt, tr, match_distance=MATCH_D)
        assert m['MOTA'] == pytest.approx(1.0)
        assert m['False Positives'] == 0
        assert m['False Negatives'] == 0
        assert m['ID Switches'] == 0
        assert m['Total Matches'] == 2

    def test_all_missed(self):
        gt = {0: [_gt(1, 0, 0)]}
        tr = {0: []}
        m = compute_metrics(gt, tr, match_distance=MATCH_D)
        assert m['False Negatives'] == 1
        assert m['Total Matches'] == 0
        assert m['MOTA'] < 0 or m['MOTA'] == pytest.approx(0.0, abs=1e-9)

    def test_all_false_positives(self):
        gt = {0: []}
        tr = {0: [_trk(1, 100, 100)]}
        m = compute_metrics(gt, tr, match_distance=MATCH_D)
        assert m['False Positives'] == 1
        assert m['Total GT'] == 0

    def test_too_far_not_matched(self):
        gt = {0: [_gt(1, 0, 0)]}
        tr = {0: [_trk(10, 50, 50)]}   # way beyond match_distance
        m = compute_metrics(gt, tr, match_distance=MATCH_D)
        assert m['False Negatives'] == 1
        assert m['False Positives'] == 1
        assert m['Total Matches'] == 0

    def test_id_switch_detected(self):
        gt = {0: [_gt(1, 0, 0)], 1: [_gt(1, 1, 0)]}
        # frame 0 matched to track 10, frame 1 matched to track 99 → 1 ID switch
        tr = {0: [_trk(10, 0, 0)], 1: [_trk(99, 1, 0)]}
        m = compute_metrics(gt, tr, match_distance=MATCH_D)
        assert m['ID Switches'] == 1

    def test_motp_is_mean_distance(self):
        gt = {0: [_gt(1, 0, 0)]}
        tr = {0: [_trk(10, 1, 0)]}   # center dist = 1.0m (z+h/2 offset cancels)
        m = compute_metrics(gt, tr, match_distance=MATCH_D)
        # GT center = (0, 0, 0.5), track center = (1, 0, 0.5) → dist = 1.0
        assert m['MOTP (mean center dist m)'] == pytest.approx(1.0, abs=1e-6)

    def test_empty_sequences(self):
        m = compute_metrics({}, {}, match_distance=MATCH_D)
        assert m['MOTA'] == pytest.approx(0.0)
        assert m['Total GT'] == 0


# ── compute_hota ──────────────────────────────────────────────────────────────

class TestComputeHota:
    def test_perfect_tracking_gives_hota_one(self):
        gt = {i: [_gt(1, float(i), 0)] for i in range(5)}
        tr = {i: [_trk(10, float(i), 0)] for i in range(5)}
        h = compute_hota(gt, tr)
        assert h['HOTA'] == pytest.approx(1.0, abs=1e-6)
        assert h['DetA'] == pytest.approx(1.0, abs=1e-6)
        assert h['AssA'] == pytest.approx(1.0, abs=1e-6)

    def test_no_detections_gives_zero(self):
        gt = {0: [_gt(1, 0, 0)]}
        tr = {0: []}
        h = compute_hota(gt, tr)
        assert h['HOTA'] == pytest.approx(0.0)
        assert h['DetA'] == pytest.approx(0.0)

    def test_all_fp_gives_zero_deta(self):
        gt = {0: []}
        tr = {0: [_trk(10, 0, 0)]}
        h = compute_hota(gt, tr)
        assert h['DetA'] == pytest.approx(0.0)

    def test_id_switch_degrades_assa(self):
        # One GT track matched to two different predicted tracks → AssA < 1
        gt = {0: [_gt(1, 0, 0)], 1: [_gt(1, 1, 0)], 2: [_gt(1, 2, 0)]}
        tr = {0: [_trk(10, 0, 0)], 1: [_trk(99, 1, 0)], 2: [_trk(10, 2, 0)]}
        h_switch = compute_hota(gt, tr)

        # Compare against no ID switch
        tr_clean = {0: [_trk(10, 0, 0)], 1: [_trk(10, 1, 0)], 2: [_trk(10, 2, 0)]}
        h_clean = compute_hota(gt, tr_clean)

        assert h_switch['AssA'] < h_clean['AssA']

    def test_multiple_gt_tracks(self):
        gt = {
            0: [_gt(1, 0, 0), _gt(2, 10, 0)],
            1: [_gt(1, 1, 0), _gt(2, 11, 0)],
        }
        tr = {
            0: [_trk(10, 0, 0), _trk(20, 10, 0)],
            1: [_trk(10, 1, 0), _trk(20, 11, 0)],
        }
        h = compute_hota(gt, tr)
        assert h['HOTA'] == pytest.approx(1.0, abs=1e-6)

    def test_hota_is_geometric_mean_of_deta_assa(self):
        gt = {0: [_gt(1, 0, 0)], 1: [_gt(1, 1, 0)]}
        tr = {0: [_trk(10, 0, 0), _trk(20, 5, 0)], 1: [_trk(10, 1, 0)]}
        h = compute_hota(gt, tr, alpha_vals=np.array([1.0]))
        assert h['HOTA'] == pytest.approx(np.sqrt(h['DetA'] * h['AssA']), abs=1e-6)


# ── compute_mt_pt_ml ──────────────────────────────────────────────────────────

class TestComputeMtPtMl:
    def test_fully_tracked_is_mt(self):
        # GT track appears 5 frames, matched all 5 → MT
        gt = {i: [_gt(1, float(i), 0)] for i in range(5)}
        tr = {i: [_trk(10, float(i), 0)] for i in range(5)}
        r = compute_mt_pt_ml(gt, tr, match_distance=MATCH_D)
        assert r['MT'] == 1
        assert r['PT'] == 0
        assert r['ML'] == 0
        assert r['MT%'] == pytest.approx(1.0)

    def test_never_tracked_is_ml(self):
        gt = {i: [_gt(1, float(i), 0)] for i in range(5)}
        tr = {i: [] for i in range(5)}
        r = compute_mt_pt_ml(gt, tr, match_distance=MATCH_D)
        assert r['ML'] == 1
        assert r['MT'] == 0

    def test_partial_tracking_is_pt(self):
        # GT appears 10 frames, matched in 5 (50%) → PT
        gt = {i: [_gt(1, float(i), 0)] for i in range(10)}
        tr = {i: ([_trk(10, float(i), 0)] if i < 5 else []) for i in range(10)}
        r = compute_mt_pt_ml(gt, tr, match_distance=MATCH_D)
        assert r['PT'] == 1
        assert r['MT'] == 0
        assert r['ML'] == 0

    def test_counts_sum_to_total_tracks(self):
        gt = {
            0: [_gt(1, 0, 0), _gt(2, 20, 0)],
            1: [_gt(1, 1, 0), _gt(2, 21, 0)],
        }
        tr = {
            0: [_trk(10, 0, 0)],   # matches GT 1 only
            1: [_trk(10, 1, 0)],
        }
        r = compute_mt_pt_ml(gt, tr, match_distance=MATCH_D)
        assert r['MT'] + r['PT'] + r['ML'] == r['GT Tracks']

    def test_empty_input(self):
        r = compute_mt_pt_ml({}, {}, match_distance=MATCH_D)
        assert r['GT Tracks'] == 0
        assert r['MT%'] == pytest.approx(0.0)


# ── compute_idf1 ──────────────────────────────────────────────────────────────

class TestComputeIdf1:
    def test_perfect_match_gives_one(self):
        gt = {0: [_gt(1, 0, 0)], 1: [_gt(1, 1, 0)]}
        tr = {0: [_trk(10, 0, 0)], 1: [_trk(10, 1, 0)]}
        r = compute_idf1(gt, tr, match_distance=MATCH_D)
        assert r['IDF1'] == pytest.approx(1.0, abs=1e-6)

    def test_no_detections_gives_zero(self):
        gt = {0: [_gt(1, 0, 0)]}
        tr = {0: []}
        r = compute_idf1(gt, tr, match_distance=MATCH_D)
        assert r['IDF1'] == pytest.approx(0.0)

    def test_id_switch_reduces_idf1(self):
        gt = {0: [_gt(1, 0, 0)], 1: [_gt(1, 1, 0)], 2: [_gt(1, 2, 0)]}
        tr_clean  = {i: [_trk(10, float(i), 0)] for i in range(3)}
        tr_switch = {0: [_trk(10, 0, 0)], 1: [_trk(99, 1, 0)], 2: [_trk(10, 2, 0)]}
        idf1_clean  = compute_idf1(gt, tr_clean,  match_distance=MATCH_D)['IDF1']
        idf1_switch = compute_idf1(gt, tr_switch, match_distance=MATCH_D)['IDF1']
        assert idf1_switch < idf1_clean

    def test_empty_sequences(self):
        r = compute_idf1({}, {}, match_distance=MATCH_D)
        assert r['IDF1'] == pytest.approx(0.0)


# ── compute_per_class ─────────────────────────────────────────────────────────

class TestComputePerClass:
    def test_classes_are_split_correctly(self):
        gt = {0: [_gt(1, 0, 0, cls='Car'), _gt(2, 20, 0, cls='Pedestrian')]}
        tr = {0: [_trk(10, 0, 0, cls='Car'), _trk(20, 20, 0, cls='Pedestrian')]}
        r = compute_per_class(gt, tr, match_distance=MATCH_D)
        assert 'Car' in r
        assert 'Pedestrian' in r
        assert 'Cyclist' not in r

    def test_per_class_perfect_tracking(self):
        gt = {0: [_gt(1, 0, 0, cls='Car')]}
        tr = {0: [_trk(10, 0, 0, cls='Car')]}
        r = compute_per_class(gt, tr, match_distance=MATCH_D)
        assert r['Car']['HOTA'] == pytest.approx(1.0, abs=1e-6)

    def test_cross_class_not_matched(self):
        # GT is Car, track reports Pedestrian — class filter prevents a match.
        # Per-class eval is GT-driven: only GT classes appear as keys.
        gt = {0: [_gt(1, 0, 0, cls='Car')]}
        tr = {0: [_trk(10, 0, 0, cls='Pedestrian')]}
        r = compute_per_class(gt, tr, match_distance=MATCH_D)
        assert r['Car']['False Negatives'] == 1
        assert 'Pedestrian' not in r
