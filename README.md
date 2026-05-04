# LiDAR Perception Tracker

A full-stack LiDAR perception and multi-object tracking pipeline. Supports two detection modes — classical euclidean clustering (no GPU required) and PointPillars (deep learning, PyTorch). Tracking uses SORT-3D: Kalman filter + Hungarian algorithm on 3D axis-aligned bounding boxes. Evaluated on the KITTI Tracking benchmark using MOTA and MOTP metrics.

## Architecture

Three core components:
- **Data ingestion** — loads KITTI velodyne frames and calibration, or generates synthetic scenes for testing
- **Object detection** — euclidean clustering or PointPillars, switchable via config
- **Multi-object tracking** — SORT-3D correlates detections across frames using IoU-based association

The `core/` layer is pure Python with zero ROS imports and is fully unit-testable. ROS2 nodes in `nodes/` are thin wrappers around core.

## Dependencies

**ROS2 Jazzy** (Ubuntu 24.04):
```bash
sudo apt install ros-jazzy-desktop
source /opt/ros/jazzy/setup.bash
```

**PyTorch** (required for PointPillars mode only — skip for euclidean clustering):

Install the appropriate build for your system from https://pytorch.org/get-started/locally/

**Python packages:**
```bash
pip install -r requirements.txt
```

## KITTI Dataset

Download the [KITTI Tracking benchmark](https://www.cvlibs.net/datasets/kitti/eval_tracking.php). You need:
- Left color images (optional, for visualization)
- Velodyne point clouds
- Camera calibration files
- Training labels

The dataset lives outside this repo. Set the path in `config/default.yaml`:
```yaml
data:
  source: "~/kitti_dataset/"
```

Expected directory layout:
```
kitti_dataset/
    training/
        calib/
        label_02/
        velodyne/
    testing/
        calib/
        velodyne/
```

## How to Run

```bash
# Source ROS2 first
source /opt/ros/jazzy/setup.bash

# Launch the full pipeline
ros2 launch launch/pipeline.launch.py dataset_path:=~/kitti_dataset
```

Detection mode is set in `config/default.yaml` under `detection.mode` — either `euclidean` or `pointpillars`.

## Running Tests

```bash
pytest tests/
```

## Eval Metrics

- **MOTA** (Multiple Object Tracking Accuracy): `1 - (FN + FP + IDSW) / GT` — penalizes missed detections, false positives, and identity switches
- **MOTP** (Multiple Object Tracking Precision): mean localization error across all true positive matches — measures how accurately detections are placed

IoU threshold for matching: 0.5 (configurable via `tracking.iou_threshold`).

## Future Extensions

- **CIoU loss for PointPillars** — replace the standard L1 regression loss with CIoU when training the detection head. Directly optimizes for box overlap quality rather than coordinate error, which should improve MOTA scores.
- **Semantic segmentation preprocessing** — remove non-object points (road, sidewalk, buildings) before detection rather than only removing the ground plane. Would require a pretrained model as KITTI does not provide point cloud segmentation labels.
