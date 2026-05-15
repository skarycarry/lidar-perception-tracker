"""Unit tests for preprocessing filters: range_crop, voxel_downsample, remove_ground."""
import numpy as np
import pytest

from lidar_tracker.core.preprocessing.filters import range_crop, voxel_downsample
from lidar_tracker.core.preprocessing.ground_filter import remove_ground


def _pts(*rows):
    """Build an (N,4) float32 point cloud from (x,y,z[,intensity]) tuples."""
    padded = [r if len(r) == 4 else (*r, 1.0) for r in rows]
    return np.array(padded, dtype=np.float32)


# ── range_crop ────────────────────────────────────────────────────────────────

class TestRangeCrop:
    def test_keeps_points_inside_range(self):
        pts = _pts((3, 0, 0), (10, 0, 0), (25, 0, 0))
        out = range_crop(pts, min_distance=2.0, max_distance=50.0)
        assert len(out) == 3

    def test_removes_points_closer_than_min(self):
        pts = _pts((0.5, 0, 0), (1.0, 0, 0), (5, 0, 0))
        out = range_crop(pts, min_distance=2.0, max_distance=50.0)
        assert len(out) == 1
        assert pytest.approx(out[0, 0], abs=1e-5) == 5.0

    def test_removes_points_farther_than_max(self):
        pts = _pts((5, 0, 0), (60, 0, 0), (100, 0, 0))
        out = range_crop(pts, min_distance=2.0, max_distance=50.0)
        assert len(out) == 1
        assert pytest.approx(out[0, 0], abs=1e-5) == 5.0

    def test_boundary_points_excluded(self):
        # Points at exactly min and max are excluded (strict inequalities)
        pts = _pts((2.0, 0, 0), (5.0, 0, 0), (50.0, 0, 0))
        out = range_crop(pts, min_distance=2.0, max_distance=50.0)
        assert len(out) == 1
        assert pytest.approx(out[0, 0], abs=1e-5) == 5.0

    def test_empty_input_returns_empty(self):
        pts = np.zeros((0, 4), dtype=np.float32)
        out = range_crop(pts, min_distance=2.0, max_distance=50.0)
        assert out.shape[0] == 0

    def test_all_removed_returns_empty(self):
        pts = _pts((0.1, 0, 0), (0.5, 0, 0))
        out = range_crop(pts, min_distance=2.0, max_distance=50.0)
        assert out.shape[0] == 0

    def test_3d_distance_used_not_xy_only(self):
        # Point at (0,0,5) has range 5 m — within [2,50]
        pts = _pts((0, 0, 5))
        out = range_crop(pts, min_distance=2.0, max_distance=50.0)
        assert len(out) == 1

    def test_intensity_column_preserved(self):
        pts = _pts((5, 0, 0, 0.7))
        out = range_crop(pts, min_distance=2.0, max_distance=50.0)
        assert out.shape[1] == 4
        assert pytest.approx(float(out[0, 3]), abs=1e-5) == 0.7


# ── voxel_downsample ──────────────────────────────────────────────────────────

class TestVoxelDownsample:
    def test_reduces_dense_cloud(self):
        rng = np.random.default_rng(0)
        # 1000 points clustered in a 0.1 m cube → should collapse to ~1 point at 0.2 m voxel
        pts = rng.uniform(0, 0.1, size=(1000, 3)).astype(np.float32)
        pts = np.hstack([pts, np.ones((1000, 1), dtype=np.float32)])
        out = voxel_downsample(pts, voxel_size=0.2)
        assert out.shape[0] < 10

    def test_output_has_four_columns(self):
        pts = _pts((1, 0, 0), (2, 0, 0), (3, 0, 0))
        out = voxel_downsample(pts, voxel_size=0.5)
        assert out.shape[1] == 4

    def test_spread_points_mostly_preserved(self):
        # Points 1 m apart each in their own voxel at 0.5 m voxel size
        pts = _pts(*(  (float(i), 0, 0) for i in range(10)  ))
        out = voxel_downsample(pts, voxel_size=0.5)
        assert out.shape[0] == 10

    def test_empty_input_returns_empty(self):
        pts = np.zeros((0, 4), dtype=np.float32)
        out = voxel_downsample(pts, voxel_size=0.2)
        assert out.shape[0] == 0


# ── remove_ground ─────────────────────────────────────────────────────────────

class TestRemoveGround:
    def test_flat_ground_plane_is_removed(self):
        rng = np.random.default_rng(1)
        # 200 ground points at z≈0, 50 object points well above ground
        ground = rng.uniform(-10, 10, size=(200, 2))
        ground_pts = np.hstack([ground, np.zeros((200, 1)), np.ones((200, 1))])
        above_pts  = np.hstack([
            rng.uniform(-5, 5, size=(50, 2)),
            rng.uniform(1.5, 3.0, size=(50, 1)),
            np.ones((50, 1)),
        ]).astype(np.float32)
        pts = np.vstack([ground_pts, above_pts]).astype(np.float32)

        out = remove_ground(pts, ground_threshold=0.2)
        # Most above-ground points survive; most ground points are stripped
        assert out.shape[0] < pts.shape[0]
        # z-values of remaining points should be mostly above ground
        assert float(np.median(out[:, 2])) > 0.5

    def test_no_ground_plane_keeps_all(self):
        rng = np.random.default_rng(2)
        # Scattered 3-D points — RANSAC finds a plane but few inliers
        pts = rng.uniform(-10, 10, size=(100, 4)).astype(np.float32)
        pts[:, 2] = rng.uniform(0.5, 5.0, size=100).astype(np.float32)
        out = remove_ground(pts, ground_threshold=0.05)  # very tight threshold
        # With a tight threshold very few points should be classified as ground
        assert out.shape[0] > 80

    def test_output_shape_has_four_columns(self):
        pts = np.random.default_rng(3).random((50, 4)).astype(np.float32)
        out = remove_ground(pts, ground_threshold=0.2)
        assert out.shape[1] == 4
