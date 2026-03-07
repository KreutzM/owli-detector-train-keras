# Obstacle4 E2E Results (Dataset-Integration #1)

## Dataset Source / License
- Source: Mendeley Data, DOI `10.17632/xwhnp82rhk.1`
- Title: *Obstacles Avoidance Assistance for Visually Impaired*
- License: `CC BY 4.0`
- Local raw/extracted paths:
  - `data/raw/obstacle4/obstacle4.zip` (placeholder download target)
  - `data/raw/obstacle4/extracted/` (materialized via Mendeley public API files)

## Current Verified Commands (WSL/bash, repo HEAD)
```bash
# 1) Import YOLO -> COCO (split layout train/valid supported)
python -m owli_train dataset import yolo \
  --yolo-dir data/raw/obstacle4/extracted \
  --out work/datasets/obstacle4/instances_raw.json

# 2) Normalize + BA label map
python -m owli_train dataset normalize \
  --coco work/datasets/obstacle4/instances_raw.json \
  --images-dir data/raw/obstacle4/extracted \
  --label-map configs/label_maps/obstacle4_to_ba.yaml \
  --out work/datasets/obstacle4/instances_gt.json

# 3) Validate GT COCO
python -m owli_train dataset validate \
  --coco work/datasets/obstacle4/instances_gt.json \
  --images-dir data/raw/obstacle4/extracted

# 4) COCO-critical pseudo labels (teacher, GPU)
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train dataset pseudo-label coco \
  --images-dir data/raw/obstacle4/extracted \
  --out work/datasets/obstacle4/pseudo_coco_critical.json \
  --classes person,bicycle,motorcycle,car,bus,truck \
  --score-threshold 0.45 \
  --batch-size 1

# 5) Merge GT + pseudo
python -m owli_train dataset merge coco \
  --manifest configs/merge_obstacle4_gt_pseudo.yaml \
  --out work/datasets/obstacle4/instances_combined.json

# 6) Deterministic split file used by Model Maker CSV export
#    Build the split from the merged COCO so pseudo-only classes are visible.
python -m owli_train dataset split \
  --coco work/datasets/obstacle4/instances_combined.json \
  --out-dir work/splits/obstacle4 \
  --seed 1337 \
  --ensure-train-class-coverage

# 7) Export Model Maker CSV (official split preserved: train/valid)
python -m owli_train dataset export modelmaker-csv \
  --coco work/datasets/obstacle4/instances_combined.json \
  --images-dir data/raw/obstacle4/extracted \
  --splits-json work/splits/obstacle4/splits.json \
  --out work/datasets/obstacle4/modelmaker.csv

# 8) Train EfficientDet-Lite2 (Model Maker)
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet \
  --config configs/efficientdet_lite2_obstacle4.yaml \
  --require-gpu

# 9) Inspect export
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite \
  --model work/runs/20260216-192857/artifacts/model.tflite

# 10) Quick eval (50 images)
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/datasets/obstacle4/instances_combined.json \
  --images-dir data/raw/obstacle4/extracted \
  --model work/runs/20260216-192857/artifacts/model.tflite \
  --limit-images 50 \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --out work/runs/20260216-192857/reports/eval_efficientdet_tflite_50_noise.json

# 11) Golden sample
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect \
  --model work/runs/20260216-192857/artifacts/model.tflite \
  --image data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg \
  --out work/runs/20260216-192857/reports/golden_obstacle4.json \
  --score-threshold 0.1 \
  --max-results 20
```

Current gate status on repo HEAD:
- The previous `score-threshold 0.6` pseudo-label recipe produced `0` `bus` detections, so `bus` disappeared before the split step and the EfficientDet gate correctly failed.
- A verified threshold sweep on repo HEAD showed that `score-threshold 0.45` keeps `4` `bus` detections, with `2` already landing in `TRAIN` under seed `1337`.
- The current recommended path therefore uses `score-threshold 0.45` plus `dataset split --ensure-train-class-coverage` on `instances_combined.json`.

## Artifacts
- Run directory: `work/runs/20260216-192857`
- Trained TFLite model: `work/runs/20260216-192857/artifacts/model.tflite`
- Combined COCO: `work/datasets/obstacle4/instances_combined.json`
- Merge report: `work/datasets/obstacle4/instances_combined.report.json`
- Eval report: `work/runs/20260216-192857/reports/eval_efficientdet_tflite_50_noise.json`
- Golden sample: `work/runs/20260216-192857/reports/golden_obstacle4.json`

## Pseudo-Label Threshold Check
- Verified source run: `work/datasets/obstacle4/pseudo_coco_critical_t03.json`
- Images processed: `1250`
- Detections kept at `score-threshold 0.3`: `530`
- Filtered class counts at the production threshold `0.45`:
  - `person`: `46`
  - `bicycle`: `10`
  - `motorcycle`: `7`
  - `car`: `209`
  - `bus`: `4`
  - `truck`: `12`

## Merge Summary
- Images: `1250`
- Annotations: `1780`
- Categories: `10` (4 BA + 6 COCO-critical)
- One pseudo annotation was suppressed by GT overlap (`pseudo_overlap_gt=1`).

## Eval Summary (50 images, score threshold 0.1)
- AP@[0.50:0.95]: `0.456`
- AP50: `0.636`
- AP75: `0.562`
- BA-class detections are non-zero (e.g. `obstacle_pole`: predictions `204`, TP `51`).
- FP/100 images:
  - threshold `0.05`: `2194.0`
  - threshold `0.10`: `896.0`
  - threshold `0.30`: `24.0`

## Golden Output
- File: `work/runs/20260216-192857/reports/golden_obstacle4.json`
- Detections written: `14`
