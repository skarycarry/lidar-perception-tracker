"""Unit tests for Sort3D tracker (track lifecycle, confirmation, deletion, duplicates)."""
import numpy as np
import pytest

from lidar_tracker.core.detection.base import Detection
from lidar_tracker.core.tracking.sort3d import Sort3D


def _det(x, y, z=0.0, h=1.0, w=1.0, l=2.0, cls='Car'):
    return Detection(x=x, y=y, z=z, height=h, width=w, length=l,
                     rotation_y=0.0, confidence=1.0, object_type=cls)


def _feed(tracker, detections, n_frames=1, dt=0.1):
    """Push identical detection list into the tracker for n_frames."""
    result = []
    for _ in range(n_frames):
        result = tracker.update(list(detections), dt=dt)
    return result


# ── Track confirmation ────────────────────────────────────────────────────────

class TestConfirmation:
    def test_not_confirmed_before_min_hits(self):
        trk = Sort3D(max_age=5, min_hits=3, match_distance=2.0)
        det = _det(0, 0)
        # Two updates → hits=2, still below min_hits=3
        trk.update([det], dt=0.1)
        out = trk.update([det], dt=0.1)
        assert len(out) == 0

    def test_confirmed_at_min_hits(self):
        trk = Sort3D(max_age=5, min_hits=3, match_distance=2.0)
        det = _det(0, 0)
        for _ in range(3):
            out = trk.update([det], dt=0.1)
        assert len(out) == 1

    def test_min_hits_one_confirms_immediately(self):
        trk = Sort3D(max_age=5, min_hits=1, match_distance=2.0)
        out = trk.update([_det(0, 0)], dt=0.1)
        assert len(out) == 1

    def test_track_id_is_stable_across_frames(self):
        trk = Sort3D(max_age=5, min_hits=3, match_distance=2.0)
        det = _det(0, 0)
        ids = []
        for _ in range(5):
            out = trk.update([det], dt=0.1)
            if out:
                ids.append(out[0].track_id)
        assert len(set(ids)) == 1


# ── Track deletion ────────────────────────────────────────────────────────────

class TestDeletion:
    def test_track_deleted_after_max_age(self):
        trk = Sort3D(max_age=2, min_hits=1, match_distance=2.0)
        # Confirm the track
        for _ in range(3):
            trk.update([_det(0, 0)], dt=0.1)
        # Stop sending detections — track should vanish after max_age+1 predict steps
        for _ in range(3):
            trk.update([], dt=0.1)
        assert len(trk.tracks) == 0

    def test_track_survives_within_max_age(self):
        trk = Sort3D(max_age=3, min_hits=1, match_distance=2.0)
        trk.update([_det(0, 0)], dt=0.1)
        # Miss for max_age frames exactly — track is at the boundary, not yet deleted
        for _ in range(3):
            trk.update([], dt=0.1)
        assert len(trk.tracks) == 1

    def test_revived_track_gets_new_id(self):
        trk = Sort3D(max_age=1, min_hits=1, match_distance=2.0)
        out1 = trk.update([_det(0, 0)], dt=0.1)
        trk.update([], dt=0.1)
        trk.update([], dt=0.1)   # now deleted
        out2 = trk.update([_det(0, 0)], dt=0.1)
        # Both confirmed (min_hits=1); different track IDs
        assert out1[0].track_id != out2[0].track_id


# ── Multi-track scenarios ─────────────────────────────────────────────────────

class TestMultiTrack:
    def test_two_far_apart_tracks(self):
        trk = Sort3D(max_age=5, min_hits=1, match_distance=2.0)
        out = trk.update([_det(0, 0), _det(100, 0)], dt=0.1)
        assert len(out) == 2
        assert out[0].track_id != out[1].track_id

    def test_new_detection_spawns_new_track(self):
        trk = Sort3D(max_age=5, min_hits=1, match_distance=2.0)
        trk.update([_det(0, 0)], dt=0.1)
        out = trk.update([_det(0, 0), _det(100, 0)], dt=0.1)
        assert len(out) == 2

    def test_count_correct_over_multiple_frames(self):
        trk = Sort3D(max_age=5, min_hits=2, match_distance=2.0)
        dets = [_det(0, 0), _det(50, 0), _det(100, 0)]
        for _ in range(2):
            out = trk.update(list(dets), dt=0.1)
        assert len(out) == 3


# ── Duplicate suppression ─────────────────────────────────────────────────────

class TestDuplicateSuppression:
    def test_duplicate_within_match_distance_is_suppressed(self):
        # Start with two separate tracks, then bring one very close to the other.
        # The track with more hits should survive.
        trk = Sort3D(max_age=10, min_hits=1, match_distance=3.0)

        # Frame 1: two tracks far apart — both confirm (min_hits=1)
        trk.update([_det(0, 0), _det(20, 0)], dt=0.1)

        # Frame 2: move second detection close to first
        out = trk.update([_det(0, 0), _det(0.5, 0)], dt=0.1)

        # Confirmed tracks whose Kalman states are within match_distance of each
        # other should be suppressed to at most 1 per cluster.
        positions = np.array([t.state[:3] for t in out])
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                assert np.linalg.norm(positions[i] - positions[j]) >= trk.match_distance

    def test_no_suppression_when_far_apart(self):
        trk = Sort3D(max_age=5, min_hits=1, match_distance=2.0)
        out = trk.update([_det(0, 0), _det(10, 0)], dt=0.1)
        assert len(out) == 2


# ── Kalman prediction ─────────────────────────────────────────────────────────

class TestKalmanPrediction:
    def test_predicted_state_tracks_moving_object(self):
        trk = Sort3D(max_age=5, min_hits=1, match_distance=5.0)
        # Feed linearly moving object: x increases by 1 each frame
        for i in range(5):
            trk.update([_det(float(i), 0)], dt=1.0)

        # After 5 updates the Kalman filter should have a positive x-velocity
        assert trk.tracks[0].state[3] > 0   # vx > 0

    def test_stationary_object_stays_near_origin(self):
        trk = Sort3D(max_age=5, min_hits=1, match_distance=2.0)
        for _ in range(5):
            trk.update([_det(0.0, 0.0)], dt=0.1)
        state = trk.tracks[0].state
        assert abs(state[0]) < 0.5
        assert abs(state[1]) < 0.5
