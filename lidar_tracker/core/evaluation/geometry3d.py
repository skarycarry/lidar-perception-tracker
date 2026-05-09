from typing import Protocol
import numpy as np


class RotatedBox(Protocol):
    x: float
    y: float
    z: float
    height: float
    width: float
    length: float
    rotation_y: float


def _corners_2d(box: RotatedBox) -> list[tuple[float, float]]:
    """4 corners of the box footprint in the XY plane."""
    cos_r = np.cos(box.rotation_y)
    sin_r = np.sin(box.rotation_y)
    hl, hw = box.length / 2.0, box.width / 2.0
    offsets = [
        ( hl * cos_r - hw * sin_r,  hl * sin_r + hw * cos_r),
        (-hl * cos_r - hw * sin_r, -hl * sin_r + hw * cos_r),
        (-hl * cos_r + hw * sin_r, -hl * sin_r - hw * cos_r),
        ( hl * cos_r + hw * sin_r,  hl * sin_r - hw * cos_r),
    ]
    return [(box.x + dx, box.y + dy) for dx, dy in offsets]


def _clip_polygon(subject: list, clip: list) -> list:
    """Sutherland-Hodgman polygon clipping of subject against clip polygon."""
    def _inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def _intersect(p1, p2, a, b):
        d1 = (p2[0] - p1[0], p2[1] - p1[1])
        d2 = (b[0] - a[0],   b[1] - a[1])
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-10:
            return p1
        t = ((a[0] - p1[0]) * d2[1] - (a[1] - p1[1]) * d2[0]) / denom
        return (p1[0] + t * d1[0], p1[1] + t * d1[1])

    output = list(subject)
    n = len(clip)
    for i in range(n):
        if not output:
            return []
        a, b = clip[i], clip[(i + 1) % n]
        incoming = output
        output = []
        for j in range(len(incoming)):
            curr = incoming[j]
            prev = incoming[j - 1]
            if _inside(curr, a, b):
                if not _inside(prev, a, b):
                    output.append(_intersect(prev, curr, a, b))
                output.append(curr)
            elif _inside(prev, a, b):
                output.append(_intersect(prev, curr, a, b))
    return output


def _polygon_area(pts: list) -> float:
    n = len(pts)
    if n < 3:
        return 0.0
    area = sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
        for i in range(n)
    )
    return abs(area) / 2.0


def iou_3d(box_a: RotatedBox, box_b: RotatedBox) -> float:
    """Rotated 3D IoU: BEV polygon intersection × height overlap."""
    az_min, az_max = box_a.z, box_a.z + box_a.height
    bz_min, bz_max = box_b.z, box_b.z + box_b.height
    h_overlap = max(0.0, min(az_max, bz_max) - max(az_min, bz_min))
    if h_overlap == 0.0:
        return 0.0

    corners_a = _corners_2d(box_a)
    corners_b = _corners_2d(box_b)
    inter_poly = _clip_polygon(corners_a, corners_b)
    inter_area = _polygon_area(inter_poly)

    inter_vol = inter_area * h_overlap
    vol_a = box_a.length * box_a.width * box_a.height
    vol_b = box_b.length * box_b.width * box_b.height
    union_vol = vol_a + vol_b - inter_vol

    if union_vol <= 0.0:
        return 0.0
    return inter_vol / union_vol
