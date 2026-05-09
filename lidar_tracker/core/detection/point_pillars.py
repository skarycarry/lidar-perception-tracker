import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from .base import Detection, Detector

# ── Config (matches OpenPCDet kitti_models/pointpillar.yaml) ─────────────────
PC_RANGE     = [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
VOXEL_SIZE   = [0.16, 0.16, 4.0]
MAX_POINTS   = 32
MAX_VOXELS   = 40_000
FEAT_STRIDE  = 2          # feature map is 2× downsampled vs. BEV grid

# 3 classes × 2 rotations = 6 anchor types per grid cell
# [l, w, h, z_bottom]  — z_bottom from OpenPCDet KITTI config
ANCHOR_CONFIGS = [
    {'class_name': 'Car',        'size': [3.9,  1.6,  1.56], 'z_bottom': -1.78, 'class_idx': 0},
    {'class_name': 'Pedestrian', 'size': [0.8,  0.6,  1.73], 'z_bottom': -0.60, 'class_idx': 1},
    {'class_name': 'Cyclist',    'size': [1.76, 0.6,  1.73], 'z_bottom': -0.60, 'class_idx': 2},
]
ANCHOR_ROTATIONS = [0.0, np.pi / 2]
CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']
DIR_OFFSET  = 0.78539  # π/4 — from OpenPCDet dir classifier


# ── Pillarization ─────────────────────────────────────────────────────────────

def _pillarize(points: np.ndarray):
    """Convert (N,4) point cloud to pillar tensors ready for PillarVFE."""
    x_min, y_min, z_min, x_max, y_max, z_max = PC_RANGE
    dx, dy, dz = VOXEL_SIZE
    nx = round((x_max - x_min) / dx)   # 432
    ny = round((y_max - y_min) / dy)   # 496

    # Filter to PC range
    m = ((points[:, 0] >= x_min) & (points[:, 0] < x_max) &
         (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
         (points[:, 2] >= z_min) & (points[:, 2] < z_max))
    pts = points[m]
    if pts.shape[0] == 0:
        dummy_feats = np.zeros((1, MAX_POINTS, 10), dtype=np.float32)
        dummy_coords = np.zeros((1, 4), dtype=np.int32)
        dummy_counts = np.array([0], dtype=np.int32)
        return dummy_feats, dummy_coords, dummy_counts

    xi = np.clip(((pts[:, 0] - x_min) / dx).astype(np.int32), 0, nx - 1)
    yi = np.clip(((pts[:, 1] - y_min) / dy).astype(np.int32), 0, ny - 1)
    pillar_idx = yi * nx + xi

    order = np.argsort(pillar_idx, kind='stable')
    pillar_idx = pillar_idx[order]
    pts = pts[order]

    unique, counts = np.unique(pillar_idx, return_counts=True)
    if len(unique) > MAX_VOXELS:
        keep = np.sort(np.random.choice(len(unique), MAX_VOXELS, replace=False))
        unique = unique[keep]
        counts = counts[keep]

    n_pillars = len(unique)
    voxel_features = np.zeros((n_pillars, MAX_POINTS, 10), dtype=np.float32)
    voxel_coords   = np.zeros((n_pillars, 4), dtype=np.int32)   # [batch, z, y, x]
    voxel_counts   = np.zeros(n_pillars, dtype=np.int32)

    split = np.searchsorted(pillar_idx, unique)
    z_offset = dz / 2.0 + z_min   # −1.0 for KITTI config

    for i in range(n_pillars):
        pidx  = unique[i]
        start = split[i]
        cnt   = counts[i]
        n     = min(cnt, MAX_POINTS)
        p     = pts[start:start + n]              # (n, 4)

        voxel_features[i, :n, :4] = p            # x, y, z, intensity

        mean_xyz = p[:, :3].mean(axis=0)
        voxel_features[i, :n, 4] = p[:, 0] - mean_xyz[0]   # x relative to cluster
        voxel_features[i, :n, 5] = p[:, 1] - mean_xyz[1]
        voxel_features[i, :n, 6] = p[:, 2] - mean_xyz[2]

        pix_x = int(pidx % nx)
        pix_y = int(pidx // nx)
        cx = x_min + (pix_x + 0.5) * dx
        cy = y_min + (pix_y + 0.5) * dy
        voxel_features[i, :n, 7] = p[:, 0] - cx            # x relative to pillar centre
        voxel_features[i, :n, 8] = p[:, 1] - cy
        voxel_features[i, :n, 9] = p[:, 2] - z_offset      # z relative to grid bottom-centre

        voxel_coords[i] = [0, 0, pix_y, pix_x]
        voxel_counts[i] = n

    return voxel_features, voxel_coords, voxel_counts


# ── Network modules ───────────────────────────────────────────────────────────

class PillarVFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.pfn_layers = nn.ModuleList([_PFNLayer(10, 64)])

    def forward(self, features, voxel_counts):
        # features: (P, max_pts, 10)
        # mask out padding entries
        mask = torch.arange(MAX_POINTS, device=features.device).unsqueeze(0) < voxel_counts.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()   # (P, max_pts, 1)
        for layer in self.pfn_layers:
            features = layer(features, mask)
        return features  # (P, 64)


class _PFNLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch, bias=False)
        self.norm   = nn.BatchNorm1d(out_ch, eps=1e-3, momentum=0.01)

    def forward(self, x, mask):
        # x: (P, T, C)
        P, T, C = x.shape
        x = self.linear(x.view(P * T, C)).view(P, T, -1)   # (P, T, out)
        x = self.norm(x.view(P * T, -1)).view(P, T, -1)
        x = torch.relu(x)
        x = x * mask                                         # zero padding
        return x.max(dim=1).values                          # (P, out)


class PointPillarScatter(nn.Module):
    def __init__(self):
        super().__init__()
        nx = round((PC_RANGE[3] - PC_RANGE[0]) / VOXEL_SIZE[0])  # 432
        ny = round((PC_RANGE[4] - PC_RANGE[1]) / VOXEL_SIZE[1])  # 496
        self.nx, self.ny = nx, ny

    def forward(self, pillar_feats, coords):
        # pillar_feats: (P, 64),  coords: (P, 4) [batch, z, y, x]
        canvas = torch.zeros(1, 64, self.ny, self.nx,
                             device=pillar_feats.device, dtype=pillar_feats.dtype)
        canvas[coords[:, 0], :, coords[:, 2], coords[:, 3]] = pillar_feats
        return canvas  # (1, 64, 496, 432)


class BaseBEVBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = [
            # (in_ch, out_ch, stride, n_repeat)
            (64,  64,  2, 3),
            (64,  128, 2, 5),
            (128, 256, 2, 5),
        ]
        self.blocks = nn.ModuleList()
        for in_ch, out_ch, stride, n in cfg:
            layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for _ in range(n):
                layers += [
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ]
            self.blocks.append(nn.Sequential(*layers))

        # Deblocks: (in_ch, out_ch, kernel, stride)
        deblock_cfg = [(64, 128, 1, 1), (128, 128, 2, 2), (256, 128, 4, 4)]
        self.deblocks = nn.ModuleList()
        for in_ch, out_ch, k, s in deblock_cfg:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, k, stride=s, bias=False),
                nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

    def forward(self, x):
        outs = []
        for block, deblock in zip(self.blocks, self.deblocks):
            x = block(x)
            outs.append(deblock(x))
        return torch.cat(outs, dim=1)  # (1, 384, 248, 216)


class AnchorHeadSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_cls     = nn.Conv2d(384, 18, 1)
        self.conv_box     = nn.Conv2d(384, 42, 1)
        self.conv_dir_cls = nn.Conv2d(384, 12, 1)

    def forward(self, x):
        return self.conv_cls(x), self.conv_box(x), self.conv_dir_cls(x)


class PointPillarsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe            = PillarVFE()
        self.scatter        = PointPillarScatter()
        self.backbone_2d    = BaseBEVBackbone()
        self.dense_head     = AnchorHeadSingle()

    def forward(self, voxel_feats, coords, voxel_counts):
        x = self.vfe(voxel_feats, voxel_counts)       # (P, 64)
        x = self.scatter(x, coords)                   # (1, 64, 496, 432)
        x = self.backbone_2d(x)                        # (1, 384, 248, 216)
        return self.dense_head(x)                      # cls, box, dir


# ── Anchor generation & box decoding ─────────────────────────────────────────

def _build_anchors():
    x_min, y_min = PC_RANGE[0], PC_RANGE[1]
    dx = VOXEL_SIZE[0] * FEAT_STRIDE   # 0.32
    dy = VOXEL_SIZE[1] * FEAT_STRIDE
    nx = round((PC_RANGE[3] - PC_RANGE[0]) / dx)  # 216
    ny = round((PC_RANGE[4] - PC_RANGE[1]) / dy)  # 248

    xs = x_min + np.arange(nx) * dx   # align_center=False
    ys = y_min + np.arange(ny) * dy
    xg, yg = np.meshgrid(xs, ys, indexing='ij')   # (216, 248)

    anchor_list  = []
    class_labels = []
    for cfg in ANCHOR_CONFIGS:
        l, w, h = cfg['size']
        z_ctr = cfg['z_bottom'] + h / 2.0
        for rot in ANCHOR_ROTATIONS:
            a = np.stack([
                xg, yg,
                np.full_like(xg, z_ctr),
                np.full_like(xg, l),
                np.full_like(xg, w),
                np.full_like(xg, h),
                np.full_like(xg, rot),
            ], axis=-1)                              # (216, 248, 7)
            anchor_list.append(a)
            class_labels.append(cfg['class_idx'])

    # Stack: (6, 216, 248, 7) → rearrange to match head output (H=248, W=216)
    anchors = np.stack(anchor_list, axis=0)          # (6, 216, 248, 7)
    anchors = anchors.transpose(0, 2, 1, 3)          # (6, 248, 216, 7)  H×W
    return anchors, class_labels                      # class_labels: 6-element list


def _decode_boxes(pred, anchor):
    """ResidualCoder inverse: pred/anchor both (..., 7)."""
    xa, ya, za = anchor[..., 0], anchor[..., 1], anchor[..., 2]
    la, wa, ha, ra = anchor[..., 3], anchor[..., 4], anchor[..., 5], anchor[..., 6]
    dx, dy, dz = pred[..., 0], pred[..., 1], pred[..., 2]
    dl, dw, dh, dr = pred[..., 3], pred[..., 4], pred[..., 5], pred[..., 6]

    diag = torch.sqrt(la ** 2 + wa ** 2)
    x = dx * diag + xa
    y = dy * diag + ya
    z = dz * ha  + za
    l = torch.exp(dl.clamp(max=4)) * la
    w = torch.exp(dw.clamp(max=4)) * wa
    h = torch.exp(dh.clamp(max=4)) * ha
    r = dr + ra
    return torch.stack([x, y, z, l, w, h, r], dim=-1)


def _bev_nms(boxes_7, scores, iou_thresh):
    """Axis-aligned BEV NMS via torchvision."""
    try:
        from torchvision.ops import nms
    except ImportError:
        # fallback: keep top-100 by score
        k = min(100, scores.shape[0])
        return torch.topk(scores, k).indices

    cx, cy, l, w = boxes_7[:, 0], boxes_7[:, 1], boxes_7[:, 3], boxes_7[:, 4]
    bev = torch.stack([cx - l / 2, cy - w / 2, cx + l / 2, cy + w / 2], dim=1)
    return nms(bev, scores, iou_thresh)


# ── Public detector ───────────────────────────────────────────────────────────

class PointPillarsDetector(Detector):
    needs_external_preprocessing = False   # pillarize() filters internally

    def __init__(self, model_path: Path,
                 conf_threshold: float = 0.5,
                 nms_threshold:  float = 0.5,
                 device: str = 'cuda'):
        self.device        = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        self.nms_threshold  = nms_threshold

        self.net = PointPillarsNet()
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        self.net.load_state_dict(ckpt['model_state'], strict=False)
        self.net.to(self.device).eval()

        anchors_np, self._class_labels = _build_anchors()
        self._anchors = torch.from_numpy(anchors_np).float().to(self.device)
        # _anchors: (6, 248, 216, 7)

    @torch.no_grad()
    def detect(self, lidar_points: np.ndarray) -> list[Detection]:
        feats_np, coords_np, counts_np = _pillarize(lidar_points)

        feats  = torch.from_numpy(feats_np).float().to(self.device)
        coords = torch.from_numpy(coords_np).long().to(self.device)
        counts = torch.from_numpy(counts_np).long().to(self.device)

        cls_out, box_out, dir_out = self.net(feats, coords, counts)
        # cls_out: (1, 18, 248, 216), box_out: (1, 42, 248, 216)

        H, W = cls_out.shape[2], cls_out.shape[3]
        cls_out = cls_out[0].view(6, 3, H, W).permute(0, 2, 3, 1)   # (6, H, W, 3)
        box_out = box_out[0].view(6, 7, H, W).permute(0, 2, 3, 1)   # (6, H, W, 7)
        dir_out = dir_out[0].view(6, 2, H, W).permute(0, 2, 3, 1)   # (6, H, W, 2)

        # Collect all candidates, then NMS per class
        all_boxes  = {ci: [] for ci in range(3)}
        all_scores = {ci: [] for ci in range(3)}

        for anchor_i, cls_idx in enumerate(self._class_labels):
            scores = torch.sigmoid(cls_out[anchor_i, :, :, cls_idx])  # (H, W)
            mask   = scores > self.conf_threshold
            if mask.sum() == 0:
                continue

            s    = scores[mask]
            pred = box_out[anchor_i][mask]
            anch = self._anchors[anchor_i][mask]
            ddir = dir_out[anchor_i][mask]

            boxes = _decode_boxes(pred, anch)            # (K, 7)

            dir_cls = ddir.argmax(dim=-1).bool()
            heading = boxes[:, 6].clone()
            heading_shifted = heading - DIR_OFFSET
            flip = dir_cls & (heading_shifted.sin() < 0)
            heading[flip] += np.pi
            boxes[:, 6] = heading

            all_boxes[cls_idx].append(boxes)
            all_scores[cls_idx].append(s)

        detections = []
        for cls_idx in range(3):
            if not all_boxes[cls_idx]:
                continue
            boxes  = torch.cat(all_boxes[cls_idx],  dim=0)
            scores = torch.cat(all_scores[cls_idx], dim=0)
            keep   = _bev_nms(boxes, scores, self.nms_threshold)
            for ki in keep:
                b = boxes[ki]
                h = float(b[5])
                detections.append(Detection(
                    x=float(b[0]), y=float(b[1]),
                    z=float(b[2]) - h / 2.0,   # decode gives center z; Detection.z is bottom
                    length=float(b[3]), width=float(b[4]), height=h,
                    rotation_y=float(b[6]),
                    confidence=float(scores[ki]),
                    object_type=CLASS_NAMES[cls_idx],
                ))

        return detections
