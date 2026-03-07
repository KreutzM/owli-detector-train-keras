# Obstacle4 E2E Results (Dataset-Integration #1)

## Dataset Source / License
- Source: Mendeley Data, DOI `10.17632/xwhnp82rhk.1`
- Title: *Obstacles Avoidance Assistance for Visually Impaired*
- License: `CC BY 4.0`
- Local raw/extracted paths:
  - `data/raw/obstacle4/obstacle4.zip` (placeholder download target)
  - `data/raw/obstacle4/extracted/` (materialized via Mendeley public API files)

## Current Verified Production Flow (WSL/bash, repo HEAD)
```bash
python -m owli_train dataset import yolo \
  --yolo-dir data/raw/obstacle4/extracted \
  --out work/datasets/obstacle4/instances_raw.json

python -m owli_train dataset normalize \
  --coco work/datasets/obstacle4/instances_raw.json \
  --images-dir data/raw/obstacle4/extracted \
  --label-map configs/label_maps/obstacle4_to_ba.yaml \
  --out work/datasets/obstacle4/instances_gt.json

python -m owli_train dataset validate \
  --coco work/datasets/obstacle4/instances_gt.json \
  --images-dir data/raw/obstacle4/extracted

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train dataset pseudo-label coco \
  --images-dir data/raw/obstacle4/extracted \
  --out work/datasets/obstacle4/pseudo_coco_critical.json \
  --classes person,bicycle,motorcycle,car,bus,truck \
  --score-threshold 0.45 \
  --batch-size 1

python -m owli_train dataset merge coco \
  --manifest configs/merge_obstacle4_gt_pseudo.yaml \
  --out work/datasets/obstacle4/instances_combined.json

python -m owli_train dataset split \
  --coco work/datasets/obstacle4/instances_combined.json \
  --out-dir work/splits/obstacle4 \
  --seed 1337 \
  --ensure-train-class-coverage

python -m owli_train dataset export modelmaker-csv \
  --coco work/datasets/obstacle4/instances_combined.json \
  --images-dir data/raw/obstacle4/extracted \
  --splits-json work/splits/obstacle4/splits.json \
  --out work/datasets/obstacle4/modelmaker.csv

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet \
  --config configs/efficientdet_lite2_obstacle4.yaml \
  --run-name obstacle4-e2e-20260307 \
  --require-gpu

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite \
  --model work/runs/20260307-222245-obstacle4-e2e-20260307/artifacts/model.tflite

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/datasets/obstacle4/instances_combined.json \
  --images-dir data/raw/obstacle4/extracted \
  --model work/runs/20260307-222245-obstacle4-e2e-20260307/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 8 \
  --out work/runs/20260307-222245-obstacle4-e2e-20260307/reports/eval_efficientdet_tflite.json

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect \
  --model work/runs/20260307-222245-obstacle4-e2e-20260307/artifacts/model.tflite \
  --image data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg \
  --out work/runs/20260307-222245-obstacle4-e2e-20260307/reports/golden_obstacle4.json \
  --score-threshold 0.1 \
  --max-results 20 \
  --num-threads 8
```

## Current Verified Run (2026-03-07)
- Run directory: `work/runs/20260307-222245-obstacle4-e2e-20260307`
- Trained TFLite model: `work/runs/20260307-222245-obstacle4-e2e-20260307/artifacts/model.tflite`
- Model size: `7.1 MB`
- Export contract files:
  - `artifacts/labels.txt`
  - `artifacts/class_names.json`
  - `mapping_files.json`
- Final class contract (10 classes, preserved end-to-end):
  - `obstacle_bump`
  - `obstacle_fence`
  - `obstacle_hole`
  - `obstacle_pole`
  - `bicycle`
  - `bus`
  - `car`
  - `motorcycle`
  - `person`
  - `truck`

## Real Data Preparation Results

### Ground Truth Import / Normalize
- Imported YOLO source: `1250` images, `1627` annotations, `4` source categories.
- Normalized BA GT COCO validated successfully against `data/raw/obstacle4/extracted`.

### Pseudo Labels (`score-threshold 0.45`, batch `1`)
- Source report: `work/datasets/obstacle4/pseudo_coco_critical.report.json`
- Teacher: `https://tfhub.dev/tensorflow/efficientdet/d2/1`
- Images processed: `1250`
- Detections kept: `288`
- Runtime: `214.57 s`
- Kept detections per pseudo class:
  - `person`: `46`
  - `bicycle`: `10`
  - `motorcycle`: `7`
  - `car`: `209`
  - `bus`: `4`
  - `truck`: `12`

### Merge / Split / CSV Export
- Combined COCO: `work/datasets/obstacle4/instances_combined.json`
- Merge result: `1250` images, `1912` annotations, `10` categories.
- Pseudo labels removed by GT overlap: `3`
- Split source: `instances_combined.json`
- Split mode: `--seed 1337 --ensure-train-class-coverage`
- ModelMaker CSV: `work/datasets/obstacle4/modelmaker.csv`
- TRAIN split coverage after export:
  - `obstacle_bump`: `377`
  - `obstacle_fence`: `309`
  - `obstacle_hole`: `343`
  - `obstacle_pole`: `271`
  - `bicycle`: `10`
  - `bus`: `4`
  - `car`: `158`
  - `motorcycle`: `6`
  - `person`: `34`
  - `truck`: `9`
- Missing expected TRAIN classes: none.

## Real Training / Export Results
- Training command used `configs/efficientdet_lite2_obstacle4.yaml` with `--require-gpu` and no `--max-steps` override.
- Model Maker training completed successfully and exported `model.tflite` plus aligned label artifacts.
- `inspect tflite` result:
  - builtin ops only: `true`
  - operators: `QUANTIZE`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, `MAX_POOL_2D`, `RESIZE_NEAREST_NEIGHBOR`, `RESHAPE`, `CONCATENATION`, `LOGISTIC`, `DEQUANTIZE`, `TFLite_Detection_PostProcess`
- Final validation loss values reported at the end of training:
  - `val_det_loss`: `0.8973`
  - `val_cls_loss`: `0.5310`
  - `val_box_loss`: `0.0073`
  - `val_loss`: `0.9041`

## Real TFLite Evaluation Results (full dataset, 1250 images)
- Report JSON: `work/runs/20260307-222245-obstacle4-e2e-20260307/reports/eval_efficientdet_tflite.json`
- Report Markdown: `work/runs/20260307-222245-obstacle4-e2e-20260307/reports/eval_efficientdet_tflite.md`
- Eval settings:
  - score threshold: `0.1`
  - noise thresholds: `0.05, 0.1, 0.3`
  - CPU threads: `8`
- COCO metrics:
  - AP@[0.50:0.95]: `0.0952`
  - AP50: `0.1897`
  - AP75: `0.0886`
  - AR100: `0.1899`
- Aggregate counts:
  - TP: `1189`
  - FP: `12310`
  - FN: `723`
  - Precision: `0.0881`
  - Recall: `0.6219`
- Noise metrics:
  - threshold `0.05`: FP `29099`, FP/100 `2327.92`
  - threshold `0.10`: FP `12310`, FP/100 `984.8`
  - threshold `0.30`: FP `640`, FP/100 `51.2`

### Per-Class Aggregate on the full eval set
- `obstacle_bump`: TP `231`, FP `2934`, FN `248`, precision `0.0730`, recall `0.4823`
- `obstacle_fence`: TP `318`, FP `2869`, FN `50`, precision `0.0998`, recall `0.8641`
- `obstacle_hole`: TP `254`, FP `4264`, FN `165`, precision `0.0562`, recall `0.6062`
- `obstacle_pole`: TP `186`, FP `1074`, FN `175`, precision `0.1476`, recall `0.5152`
- `bicycle`: TP `0`, FP `0`, FN `10`, precision `0.0000`, recall `0.0000`
- `bus`: TP `0`, FP `0`, FN `4`, precision `0.0000`, recall `0.0000`
- `car`: TP `200`, FP `1168`, FN `9`, precision `0.1462`, recall `0.9569`
- `motorcycle`: TP `0`, FP `0`, FN `7`, precision `0.0000`, recall `0.0000`
- `person`: TP `0`, FP `1`, FN `43`, precision `0.0000`, recall `0.0000`
- `truck`: TP `0`, FP `0`, FN `12`, precision `0.0000`, recall `0.0000`

## Real Golden Output
- Report file: `work/runs/20260307-222245-obstacle4-e2e-20260307/reports/golden_obstacle4.json`
- Input image: `data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg`
- Detections written: `7`
- Highest-scoring outputs on this sample were mostly `obstacle_pole`, plus one `obstacle_fence` and two low-score `car` detections.
- This is technically plausible for the sample, but it does not demonstrate robust multi-class BA behavior on its own.

## Assessment
- The corrected Obstacle4 path is now reproducible end-to-end on repo HEAD and preserves the intended 10-class label contract into the final Lite2 export.
- The resulting model is a valid technical reference run for the current BA path, but not a strong product candidate yet.
- BA obstacle classes are learned to some extent, especially `obstacle_fence`, `obstacle_hole`, `obstacle_pole` and `obstacle_bump`, but precision remains low because false positives are still high.
- The COCO-critical long-tail classes remain the main weakness:
  - `car` is usable enough to detect on this dataset.
  - `person`, `bicycle`, `motorcycle`, `bus`, and `truck` are effectively not learned in the final TFLite model under this data regime.
- The run should therefore be treated as the current corrected reference baseline, not as a production-ready BA detector.

## Historical Note
- Older documentation referred to `work/runs/20260216-192857` plus a 50-image quick eval.
- That older quick check remains useful as a historical comparison point, but the current file now treats `work/runs/20260307-222245-obstacle4-e2e-20260307` as the latest fully verified Obstacle4 Lite2 reference run on repo HEAD.
