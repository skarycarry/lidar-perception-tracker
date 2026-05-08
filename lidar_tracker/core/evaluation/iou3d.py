from typing import Protocol


class BoundedBox(Protocol):
    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]: ...
    @property
    def volume(self) -> float: ...


def iou_3d(box_a: BoundedBox, box_b: BoundedBox) -> float:
    ax_min, ax_max, ay_min, ay_max, az_min, az_max = box_a.bounds
    bx_min, bx_max, by_min, by_max, bz_min, bz_max = box_b.bounds

    x_overlap = max(0, min(ax_max, bx_max) - max(ax_min, bx_min))
    y_overlap = max(0, min(ay_max, by_max) - max(ay_min, by_min))
    z_overlap = max(0, min(az_max, bz_max) - max(az_min, bz_min))

    intersection = x_overlap * y_overlap * z_overlap
    union = box_a.volume + box_b.volume - intersection
    if union == 0:
        return 0.0
    return intersection / union
