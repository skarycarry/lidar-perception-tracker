import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from lidar_tracker.core.data.kitti_loader import KittiDetection
from lidar_tracker.core.tracking.track import Track


def _center(box) -> np.ndarray:
    return np.array([box.x, box.y, box.z + box.height / 2.0])


# ── MOTA ─────────────────────────────────────────────────────────────────────

def compute_metrics(
    all_gt: dict[int, list[KittiDetection]],
    all_tracks: dict[int, list[Track]],
    match_distance: float = 2.0,
) -> dict[str, float]:
    prev_matches = {}
    total_gt = 0
    total_matches = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_id_switches = 0
    total_dist = 0.0

    for frame, gt_dets in all_gt.items():
        tracks = all_tracks.get(frame, [])
        total_gt += len(gt_dets)

        dist_matrix = np.full((len(gt_dets), len(tracks)), np.inf)
        for i, gt in enumerate(gt_dets):
            for j, track in enumerate(tracks):
                dist_matrix[i, j] = float(np.linalg.norm(
                    _center(gt) - _center(track.last_detection)
                ))

        raw = np.array(linear_sum_assignment(dist_matrix)).T
        matched_indices = [(i, j) for i, j in raw if dist_matrix[i, j] <= match_distance]

        matched_gt = set()
        matched_tracks = set()
        for gt_idx, track_idx in matched_indices:
            matched_gt.add(gt_idx)
            matched_tracks.add(track_idx)
            total_dist += dist_matrix[gt_idx, track_idx]
            if frame - 1 in prev_matches and gt_idx in prev_matches[frame - 1]:
                if prev_matches[frame - 1][gt_idx] != tracks[track_idx].track_id:
                    total_id_switches += 1

        total_matches += len(matched_gt)
        total_false_positives += len(tracks) - len(matched_tracks)
        total_false_negatives += len(gt_dets) - len(matched_gt)
        prev_matches[frame] = {
            gt_idx: tracks[track_idx].track_id for gt_idx, track_idx in matched_indices
        }

    motp = total_dist / total_matches if total_matches > 0 else 0.0
    mota = (
        1 - (total_false_positives + total_false_negatives + total_id_switches) / total_gt
        if total_gt > 0 else 0.0
    )
    return {
        "MOTA": mota,
        "MOTP (mean center dist m)": motp,
        "ID Switches": total_id_switches,
        "False Positives": total_false_positives,
        "False Negatives": total_false_negatives,
        "Total GT": total_gt,
        "Total Matches": total_matches,
    }


def weighted_mota(metrics: dict, ids_weight: float = 1.0) -> float:
    gt = metrics["Total GT"]
    if gt == 0:
        return 0.0
    return 1.0 - (
        metrics["False Positives"]
        + metrics["False Negatives"]
        + ids_weight * metrics["ID Switches"]
    ) / gt


# ── MT / PT / ML ─────────────────────────────────────────────────────────────

def compute_mt_pt_ml(
    all_gt: dict[int, list],
    all_tracks: dict[int, list],
    match_distance: float = 2.0,
) -> dict:
    """Mostly Tracked / Partially Tracked / Mostly Lost per GT trajectory."""
    gt_total:   dict[int, int] = defaultdict(int)
    gt_matched: dict[int, int] = defaultdict(int)

    for frame, gt_dets in all_gt.items():
        tracks = all_tracks.get(frame, [])
        for gt in gt_dets:
            gt_total[gt.track_id] += 1
        if not tracks:
            continue
        dist_matrix = np.full((len(gt_dets), len(tracks)), np.inf)
        for i, gt in enumerate(gt_dets):
            for j, track in enumerate(tracks):
                dist_matrix[i, j] = float(np.linalg.norm(
                    _center(gt) - _center(track.last_detection)
                ))
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        for i, j in zip(row_ind, col_ind):
            if dist_matrix[i, j] <= match_distance:
                gt_matched[gt_dets[i].track_id] += 1

    mt = ml = pt = 0
    for gt_id, total in gt_total.items():
        ratio = gt_matched.get(gt_id, 0) / total
        if ratio >= 0.8:
            mt += 1
        elif ratio <= 0.2:
            ml += 1
        else:
            pt += 1

    n = len(gt_total)
    return {
        "MT": mt, "PT": pt, "ML": ml,
        "MT%": mt / n if n > 0 else 0.0,
        "ML%": ml / n if n > 0 else 0.0,
        "GT Tracks": n,
    }


# ── IDF1 ─────────────────────────────────────────────────────────────────────

def compute_idf1(
    all_gt: dict[int, list],
    all_tracks: dict[int, list],
    match_distance: float = 2.0,
) -> dict[str, float]:
    """
    ID F1 Score.  IDF1 = 2·IDTP / (total GT detections + total pred detections).

    Bipartite-matches GT trajectories to predicted trajectories by maximising
    total true-positive association frames (TPA), then counts:
      IDTP = sum of TPA for each matched (GT, pred) pair
      IDF1 = 2·IDTP / (Σ len(g) + Σ len(p))
    """
    gt_pred_count: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    gt_len:   dict[int, int] = defaultdict(int)
    pred_len: dict[int, int] = defaultdict(int)

    for frame, gt_dets in all_gt.items():
        tracks = all_tracks.get(frame, [])
        for gt in gt_dets:
            gt_len[gt.track_id] += 1
        for t in tracks:
            pred_len[t.track_id] += 1
        if not gt_dets or not tracks:
            continue
        dist_matrix = np.full((len(gt_dets), len(tracks)), np.inf)
        for i, gt in enumerate(gt_dets):
            for j, t in enumerate(tracks):
                dist_matrix[i, j] = float(np.linalg.norm(
                    _center(gt) - _center(t.last_detection)
                ))
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        for i, j in zip(row_ind, col_ind):
            if dist_matrix[i, j] <= match_distance:
                gt_pred_count[gt_dets[i].track_id][tracks[j].track_id] += 1

    gt_ids   = list(gt_pred_count.keys())
    pred_ids = list(pred_len.keys())
    idtp = 0
    if gt_ids and pred_ids:
        pred_idx = {pid: j for j, pid in enumerate(pred_ids)}
        cost     = np.zeros((len(gt_ids), len(pred_ids)))
        for i, gid in enumerate(gt_ids):
            for pid, tpa in gt_pred_count[gid].items():
                cost[i, pred_idx[pid]] = tpa
        r, c = linear_sum_assignment(-cost)
        idtp = int(cost[r, c].sum())

    total_gt   = sum(gt_len.values())
    total_pred = sum(pred_len.values())
    denom      = total_gt + total_pred
    return {"IDF1": 2 * idtp / denom if denom > 0 else 0.0}


# ── Per-class breakdown ───────────────────────────────────────────────────────

def _split_by_class(all_gt, all_tracks):
    classes = {d.object_type for dets in all_gt.values() for d in dets}
    out = {}
    for cls in sorted(classes):
        gt_cls = {f: [d for d in dets if d.object_type == cls]
                  for f, dets in all_gt.items()}
        gt_cls = {f: d for f, d in gt_cls.items() if d}
        tr_cls = {f: [t for t in tracks if t.last_detection.object_type == cls]
                  for f, tracks in all_tracks.items()}
        out[cls] = (gt_cls, tr_cls)
    return out


def compute_per_class(
    all_gt: dict[int, list],
    all_tracks: dict[int, list],
    match_distance: float = 2.0,
) -> dict[str, dict]:
    """Return HOTA + MOTA + MT/ML + IDF1 broken down by object class."""
    out = {}
    for cls, (gt_cls, tr_cls) in _split_by_class(all_gt, all_tracks).items():
        h   = compute_hota(gt_cls, tr_cls)
        m   = compute_metrics(gt_cls, tr_cls, match_distance=match_distance)
        mt  = compute_mt_pt_ml(gt_cls, tr_cls, match_distance=match_distance)
        idf = compute_idf1(gt_cls, tr_cls, match_distance=match_distance)
        out[cls] = {**h, **m, **mt, **idf}
    return out


# ── HOTA ─────────────────────────────────────────────────────────────────────

def compute_hota(
    all_gt: dict[int, list],
    all_tracks: dict[int, list],
    alpha_vals: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Higher Order Tracking Accuracy (Luiten et al., IJCV 2021).

    Sweeps over Euclidean distance thresholds (alpha) rather than IoU thresholds,
    which is the natural adaptation for 3D LiDAR tracking.

    HOTA(α) = sqrt(DetA(α) × AssA(α))
    HOTA    = mean over α

    DetA(α) = TP / (TP + FP + FN)           — detection jaccard
    AssA(α) = (1/TP) Σ_{g,t} TPA(g,t)²     — association jaccard, averaged
                       / (TPA(g,t) + FPA(g,t) + FNA(g,t))

    where for a (GT trajectory g, predicted trajectory t) pair:
      TPA(g,t) = frames where g and t are mutually matched
      FPA(g,t) = frames where t is matched to a different GT
      FNA(g,t) = frames where g is matched to a different predicted track
    """
    if alpha_vals is None:
        # 15 thresholds from 0.5 m to 4.0 m — covers tight to loose association
        alpha_vals = np.linspace(0.5, 4.0, 15)

    hota_a, deta_a, assa_a = [], [], []
    for alpha in alpha_vals:
        h, d, a = _hota_at_alpha(all_gt, all_tracks, alpha)
        hota_a.append(h)
        deta_a.append(d)
        assa_a.append(a)

    return {
        "HOTA": float(np.mean(hota_a)),
        "DetA": float(np.mean(deta_a)),
        "AssA": float(np.mean(assa_a)),
    }


def _hota_at_alpha(all_gt, all_tracks, alpha: float):
    """Compute HOTA components at a single distance threshold."""
    # gt_pred_count[gt_id][pred_id] = number of frames they were mutually matched
    gt_pred_count: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    tpa_total = 0
    fp_det    = 0
    fn_det    = 0

    for frame, gt_dets in all_gt.items():
        tracks = all_tracks.get(frame, [])
        n_gt, n_tr = len(gt_dets), len(tracks)

        if n_gt == 0:
            fp_det += n_tr
            continue
        if n_tr == 0:
            fn_det += n_gt
            continue

        dist_matrix = np.full((n_gt, n_tr), np.inf)
        for i, gt in enumerate(gt_dets):
            for j, track in enumerate(tracks):
                dist_matrix[i, j] = float(np.linalg.norm(
                    _center(gt) - _center(track.last_detection)
                ))

        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        matched_gt = set()
        matched_tr = set()

        for i, j in zip(row_ind, col_ind):
            if dist_matrix[i, j] <= alpha:
                gt_pred_count[gt_dets[i].track_id][tracks[j].track_id] += 1
                tpa_total += 1
                matched_gt.add(i)
                matched_tr.add(j)

        fp_det += n_tr - len(matched_tr)
        fn_det += n_gt - len(matched_gt)

    if tpa_total == 0:
        return 0.0, 0.0, 0.0

    # DetA: detection jaccard over the whole sequence
    deta = tpa_total / (tpa_total + fp_det + fn_det)

    # Pre-aggregate totals for FPA/FNA calculation
    gt_total:   dict[int, int] = {}
    pred_total: dict[int, int] = defaultdict(int)
    for gt_id, pred_counts in gt_pred_count.items():
        gt_total[gt_id] = sum(pred_counts.values())
        for pred_id, count in pred_counts.items():
            pred_total[pred_id] += count

    # AssA: mean association jaccard weighted by TPA
    assa_sum = 0.0
    for gt_id, pred_counts in gt_pred_count.items():
        for pred_id, tpa in pred_counts.items():
            fpa = pred_total[pred_id] - tpa   # pred matched to other GTs
            fna = gt_total[gt_id]    - tpa    # GT matched to other preds
            assa_sum += tpa * tpa / (tpa + fpa + fna)

    assa = assa_sum / tpa_total
    hota = float(np.sqrt(deta * assa))
    return hota, deta, assa
