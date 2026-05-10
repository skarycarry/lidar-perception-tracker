import numpy as np
from scipy.spatial import KDTree


class EgoMotionEstimator:
    """
    Maintains a cumulative world pose via frame-to-frame ICP.

    Coordinate convention
    ─────────────────────
    world frame  = sensor frame of the very first call (origin fixed in space)
    sensor frame = current LiDAR frame (moves with the vehicle)

    world_T_sensor  (R_ws, t_ws):  p_world = R_ws @ p_sensor + t_ws
    sensor_T_world  (R_ws.T,  …):  p_sensor = R_ws.T @ (p_world − t_ws)

    Pose update (each frame)
    ────────────────────────
    ICP returns ego_R, ego_t  (prev_sensor → curr_sensor):
        p_curr = ego_R @ p_prev + ego_t

    Since world = prev_sensor at t=0 we accumulate:
        R_ws = R_ws_prev @ ego_R.T
        t_ws = t_ws_prev − R_ws @ ego_t        ← uses the *new* R_ws
    """

    def __init__(self, voxel_size: float = 0.8, max_iter: int = 15,
                 max_correspondence: float = 2.0):
        self.voxel_size         = voxel_size
        self.max_iter           = max_iter
        self.max_correspondence = max_correspondence

        # Cumulative world-from-sensor transform (identity at start)
        self.R_ws = np.eye(3, dtype=np.float64)
        self.t_ws = np.zeros(3, dtype=np.float64)

        self._prev_pts: np.ndarray | None = None

    # ── Public transform helpers ──────────────────────────────────────────────

    def sensor_to_world(self, pts: np.ndarray) -> np.ndarray:
        """(N,3) or (3,) sensor-frame points → world frame."""
        return (self.R_ws @ pts.T).T + self.t_ws

    def world_to_sensor(self, pts: np.ndarray) -> np.ndarray:
        """(N,3) or (3,) world-frame points → current sensor frame."""
        return (self.R_ws.T @ (pts - self.t_ws).T).T

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update(self, points: np.ndarray) -> None:
        """
        Call once per frame with the raw (N,4) point cloud.
        Updates the internal world pose; use sensor_to_world / world_to_sensor
        after the call.
        """
        curr = self._filter_and_downsample(points[:, :3])

        if self._prev_pts is None or len(curr) < 50:
            self._prev_pts = curr
            return

        ego_R, ego_t = self._icp(self._prev_pts, curr)
        self._prev_pts = curr

        # Update cumulative world pose
        self.R_ws = self.R_ws @ ego_R.T
        self.t_ws = self.t_ws - self.R_ws @ ego_t

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _filter_and_downsample(self, xyz: np.ndarray) -> np.ndarray:
        m = ((xyz[:, 0] > 2.0)  & (xyz[:, 0] < 30.0) &
             (np.abs(xyz[:, 1]) < 20.0) &
             (xyz[:, 2] > -1.8) & (xyz[:, 2] < 1.0))
        pts = xyz[m]
        if len(pts) == 0:
            return pts
        idx = np.floor(pts / self.voxel_size).astype(np.int32)
        _, unique = np.unique(idx, axis=0, return_index=True)
        return pts[unique].astype(np.float64)

    def _icp(self, source: np.ndarray, target: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
        src = source.copy()
        R_total = np.eye(3)
        t_total = np.zeros(3)
        tree = KDTree(target)

        for _ in range(self.max_iter):
            dists, idxs = tree.query(src, k=1, workers=-1)
            valid = dists < self.max_correspondence
            if valid.sum() < 20:
                break
            s = src[valid]
            t_pts = target[idxs[valid]]
            s_mean, t_mean = s.mean(0), t_pts.mean(0)
            U, _, Vt = np.linalg.svd((s - s_mean).T @ (t_pts - t_mean))
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1] *= -1
                R = Vt.T @ U.T
            t = t_mean - R @ s_mean
            src = (R @ src.T).T + t
            R_total = R @ R_total
            t_total = R @ t_total + t
            if np.linalg.norm(t) < 0.005 and np.abs(np.trace(R) - 3) < 1e-5:
                break

        return R_total, t_total
