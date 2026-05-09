# LiDAR Perception Tracker

A full-stack LiDAR perception and multi-object tracking pipeline. Supports two detection modes — classical euclidean clustering (no GPU required) and PointPillars (deep learning, PyTorch). Tracking uses SORT-3D: Kalman filter + Hungarian algorithm on 3D axis-aligned bounding boxes. Evaluated on the KITTI Tracking benchmark using MOTA and MOTP metrics.

## Architecture

Three core components:
- **Data ingestion** — loads KITTI velodyne frames and calibration, or generates synthetic scenes for testing
- **Object detection** — euclidean clustering or PointPillars, switchable via config
- **Multi-object tracking** — SORT-3D correlates detections across frames using 3D centre-distance association

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

Detection mode is set in `config/default.yaml` under `detection.mode` — either `euclidean` or `point_pillars`.

## PointPillars Pretrained Weights

Weights are not included in this repo. The recommended source is the **OpenPCDet model zoo**
(https://github.com/open-mmlab/OpenPCDet — see `docs/MODEL_ZOO.md`).

Download the KITTI `pointpillar` checkpoint and place it at:
```
models/pointpillar_7728.pth
```

The `models/` directory is git-ignored. Update `config/default.yaml` if you use a different path:
```yaml
detection:
  point_pillars:
    model_path: "models/pointpillar_7728.pth"
```

The `PointPillarsDetector` is a self-contained PyTorch re-implementation of the PointPillars
architecture (PillarVFE → PointPillarScatter → BaseBEVBackbone → AnchorHeadSingle) that loads
OpenPCDet `.pth` checkpoints directly — **no OpenPCDet or spconv installation required**.
The implementation matches OpenPCDet's layer naming convention exactly so state dicts load
without remapping.

## Running Tests

```bash
pytest tests/
```

## Eval Metrics

- **MOTA** (Multiple Object Tracking Accuracy): `1 - (FN + FP + IDSW) / GT` — penalizes missed detections, false positives, and identity switches
- **MOTP** (Multiple Object Tracking Precision): mean centre-distance (metres) of matched pairs — measures how accurately detections are localised

Matching uses 3D centre distance rather than IoU. The threshold (default 2.0 m) is configurable via `tracking.match_distance` and `evaluation.match_distance`.

## Future Extensions

- **CIoU loss for PointPillars** — replace the standard L1 regression loss with CIoU when training the detection head. Directly optimizes for box overlap quality rather than coordinate error, which should improve MOTA scores.
- **Semantic segmentation preprocessing** — remove non-object points (road, sidewalk, buildings) before detection rather than only removing the ground plane. Would require a pretrained model as KITTI does not provide point cloud segmentation labels.
