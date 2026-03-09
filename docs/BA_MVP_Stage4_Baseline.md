# BA MVP Stage-4 Replay Baseline

## Scope
- First real `EfficientDet-Lite2` / Model Maker comparison run on the Stage-4 BA-v1 dataset.
- Sources included:
  - `Obstacle4`
  - balanced `Mapillary Vistas`
  - `OD / Obstacle-Dataset`
  - small `COCO replay` for `person`, `bicycle`, `motorcycle`, `car`, `bus`, `truck`
- Goal:
  - compare the first replay-augmented run directly against the verified Stage-3 multi-source baseline
  - check whether the replay subset stabilizes rehearsal classes without materially diluting BA-core behavior

## Run Identity
- Config: [`configs/efficientdet_lite2_ba_mvp_stage4.yaml`](../configs/efficientdet_lite2_ba_mvp_stage4.yaml)
- Canonical label contract for training: [`configs/label_contracts/ba_v1.class_names.json`](../configs/label_contracts/ba_v1.class_names.json)
- Run dir:
  - `work/runs/20260308-211806-ba-mvp-stage4-20260308`
- TFLite model:
  - `work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite`
- Model size:
  - `7.1 MB`

## Data Input
- Materialized Stage-4 COCO:
  - `work/datasets/ba_mvp_stage4_with_coco_replay/instances_materialized.json`
- Materialized images:
  - `work/datasets/ba_mvp_stage4_with_coco_replay/images`
- Model Maker CSV:
  - `work/datasets/ba_mvp_stage4_with_coco_replay/modelmaker.csv`
- Split source:
  - `work/splits/ba_mvp_stage4_with_coco_replay/splits.json`

Source mix in the merged Stage-4 dataset:
- `Obstacle4`: `1250` images
- balanced `Mapillary`: `1224` images
- `OD`: `1592` images
- `COCO replay`: `785` images
- total: `4851` images, `50038` annotations, `10` categories

Held-out `TEST` split used for the native Stage-4 eval:
- images: `486`
- annotations: `5178`

## Verified Commands
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet \
  --config configs/efficientdet_lite2_ba_mvp_stage4.yaml \
  --run-name ba-mvp-stage4-20260308 \
  --require-gpu

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite \
  --model work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_mvp_stage4_with_coco_replay/instances_test.json \
  --images-dir work/datasets/ba_mvp_stage4_with_coco_replay/images \
  --model work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 8 \
  --out work/runs/20260308-211806-ba-mvp-stage4-20260308/reports/eval_efficientdet_tflite_stage4_test.json

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json \
  --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --model work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 8 \
  --out work/runs/20260308-211806-ba-mvp-stage4-20260308/reports/eval_efficientdet_tflite_stage3_test.json

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_mvp_stage4_with_coco_replay/instances_test.json \
  --images-dir work/datasets/ba_mvp_stage4_with_coco_replay/images \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 8 \
  --out work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_stage4_test.json

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect \
  --model work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite \
  --image data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg \
  --out work/runs/20260308-211806-ba-mvp-stage4-20260308/reports/golden_obstacle4.json \
  --score-threshold 0.1 \
  --max-results 20 \
  --num-threads 8
```

## Final Label Contract
The exported Lite2 artifacts preserve the canonical BA-v1 order:
1. `obstacle_bump`
2. `obstacle_fence`
3. `obstacle_hole`
4. `obstacle_pole`
5. `bicycle`
6. `bus`
7. `car`
8. `motorcycle`
9. `person`
10. `truck`

Note on older naming:
- current repo BA-v1 uses `obstacle_fence` and `obstacle_hole`
- older wording such as `obstacle_fence_rail` and `obstacle_hole_dropoff` maps to those current contract names

## Training Result
- completed epochs: `20`
- final reported losses:
  - `det_loss`: `0.8671`
  - `cls_loss`: `0.5258`
  - `box_loss`: `0.0068`
  - `loss`: `0.8745`
  - `val_det_loss`: `0.6234`
  - `val_cls_loss`: `0.3775`
  - `val_box_loss`: `0.0049`
  - `val_loss`: `0.6309`

## TFLite Inspect
- builtin ops only: `true`
- operators:
  - `QUANTIZE`
  - `CONV_2D`
  - `DEPTHWISE_CONV_2D`
  - `ADD`
  - `MAX_POOL_2D`
  - `RESIZE_NEAREST_NEIGHBOR`
  - `RESHAPE`
  - `CONCATENATION`
  - `LOGISTIC`
  - `DEQUANTIZE`
  - `TFLite_Detection_PostProcess`

## Direct Comparison Benchmarks

### 1. Common Stage-3 `TEST` split
This is the cleanest direct baseline comparison because both models are evaluated on the same held-out set from the Stage-3 dataset.

Reference:
- Stage-3 model report:
  - `work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_stage3_test.json`
- Stage-4 model on the same eval set:
  - `work/runs/20260308-211806-ba-mvp-stage4-20260308/reports/eval_efficientdet_tflite_stage3_test.json`

Global metrics:

| Metric | Stage-3 model | Stage-4 model | Delta |
| --- | ---: | ---: | ---: |
| AP@[0.50:0.95] | `0.1307` | `0.1232` | `-0.0074` |
| AP50 | `0.2325` | `0.2170` | `-0.0155` |
| AP75 | `0.1270` | `0.1203` | `-0.0067` |
| AR100 | `0.2170` | `0.2095` | `-0.0075` |
| Precision | `0.2050` | `0.2118` | `+0.0068` |
| Recall | `0.3735` | `0.3627` | `-0.0108` |
| TP | `1447` | `1405` | `-42` |
| FP | `5612` | `5229` | `-383` |
| FN | `2427` | `2469` | `+42` |
| FP/100 @ `0.05` | `2108.09` | `2099.51` | `-8.58` |
| FP/100 @ `0.10` | `1375.49` | `1281.62` | `-93.87` |
| FP/100 @ `0.30` | `111.52` | `120.59` | `+9.07` |

Per-class aggregate on the common Stage-3 `TEST` split:

| Class | Stage-3 precision | Stage-4 precision | Delta | Stage-3 recall | Stage-4 recall | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `obstacle_bump` | `0.0403` | `0.0476` | `+0.0073` | `0.5238` | `0.5000` | `-0.0238` |
| `obstacle_fence` | `0.0617` | `0.0836` | `+0.0219` | `0.2000` | `0.2783` | `+0.0783` |
| `obstacle_hole` | `0.0370` | `0.0339` | `-0.0031` | `0.3878` | `0.3469` | `-0.0408` |
| `obstacle_pole` | `0.1449` | `0.1550` | `+0.0101` | `0.1446` | `0.1473` | `+0.0027` |
| `person` | `0.2304` | `0.2549` | `+0.0245` | `0.4950` | `0.4691` | `-0.0259` |
| `bicycle` | `0.2557` | `0.2168` | `-0.0390` | `0.3223` | `0.3099` | `-0.0124` |
| `motorcycle` | `0.2784` | `0.2744` | `-0.0040` | `0.4037` | `0.3758` | `-0.0280` |
| `car` | `0.3076` | `0.3063` | `-0.0013` | `0.5701` | `0.5608` | `-0.0093` |
| `bus` | `0.3041` | `0.2756` | `-0.0286` | `0.3882` | `0.3647` | `-0.0235` |
| `truck` | `0.2217` | `0.2412` | `+0.0195` | `0.3451` | `0.2887` | `-0.0563` |

Reading:
- Stage-4 reduces low-threshold false positives somewhat, but not enough to offset the AP/recall loss.
- The only clear BA-core gains on this common set are `obstacle_fence` and a small `obstacle_pole` improvement.
- The replay classes do not stabilize overall on this set:
  - `person` precision improves, but recall falls
  - `bicycle`, `motorcycle`, `bus`, and `truck` regress
  - `car` is effectively flat to slightly worse

### 2. Common Stage-4 `TEST` split
This check asks a narrower question:
- does the Stage-4-trained model beat the old Stage-3 model on the new replay-augmented held-out set?

Reference:
- Stage-3 model on the Stage-4 eval set:
  - `work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_stage4_test.json`
- Stage-4 model on its native eval set:
  - `work/runs/20260308-211806-ba-mvp-stage4-20260308/reports/eval_efficientdet_tflite_stage4_test.json`

Global metrics:

| Metric | Stage-3 model | Stage-4 model | Delta |
| --- | ---: | ---: | ---: |
| AP@[0.50:0.95] | `0.1481` | `0.1427` | `-0.0054` |
| AP50 | `0.2729` | `0.2663` | `-0.0065` |
| AP75 | `0.1446` | `0.1389` | `-0.0057` |
| AR100 | `0.2348` | `0.2244` | `-0.0104` |
| Precision | `0.2434` | `0.2527` | `+0.0094` |
| Recall | `0.4199` | `0.4162` | `-0.0037` |
| TP | `2174` | `2155` | `-19` |
| FP | `6759` | `6372` | `-387` |
| FN | `3004` | `3023` | `+19` |
| FP/100 @ `0.05` | `2023.46` | `2012.55` | `-10.91` |
| FP/100 @ `0.10` | `1390.74` | `1311.11` | `-79.63` |
| FP/100 @ `0.30` | `140.74` | `156.38` | `+15.64` |

Per-class aggregate on the common Stage-4 `TEST` split:

| Class | Stage-3 precision | Stage-4 precision | Delta | Stage-3 recall | Stage-4 recall | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `obstacle_bump` | `0.0383` | `0.0393` | `+0.0011` | `0.6176` | `0.4706` | `-0.1471` |
| `obstacle_fence` | `0.0771` | `0.0879` | `+0.0107` | `0.2397` | `0.2810` | `+0.0413` |
| `obstacle_hole` | `0.0429` | `0.0340` | `-0.0089` | `0.3333` | `0.2639` | `-0.0694` |
| `obstacle_pole` | `0.1283` | `0.1348` | `+0.0065` | `0.1294` | `0.1371` | `+0.0077` |
| `person` | `0.3057` | `0.3375` | `+0.0318` | `0.5530` | `0.5660` | `+0.0130` |
| `bicycle` | `0.3140` | `0.2947` | `-0.0194` | `0.4402` | `0.4266` | `-0.0135` |
| `motorcycle` | `0.4058` | `0.3805` | `-0.0253` | `0.5070` | `0.4869` | `-0.0201` |
| `car` | `0.2937` | `0.3056` | `+0.0119` | `0.5765` | `0.5662` | `-0.0103` |
| `bus` | `0.3071` | `0.3034` | `-0.0037` | `0.4618` | `0.4389` | `-0.0229` |
| `truck` | `0.2719` | `0.2928` | `+0.0208` | `0.3286` | `0.3145` | `-0.0141` |

Reading:
- Even on the replay-augmented eval set, the Stage-4 model does not beat the Stage-3 model globally.
- `person` is the clearest rehearsal-class winner from replay on this set.
- `car` and `truck` gain precision but still lose recall.
- `bicycle`, `motorcycle`, and `bus` do not benefit.
- BA-core behavior remains mixed:
  - `obstacle_fence` and `obstacle_pole` improve
  - `obstacle_bump` and `obstacle_hole` get worse

## Golden Detect
- report:
  - `work/runs/20260308-211806-ba-mvp-stage4-20260308/reports/golden_obstacle4.json`
- input image:
  - `data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg`
- detections written: `14`
- class mix:
  - `obstacle_pole`: `7`
  - `obstacle_hole`: `3`
  - `obstacle_fence`: `2`
  - `car`: `2`

Reading:
- the sample is slightly less dense than the Stage-3 golden output (`14` detections vs. `20`)
- output remains obstacle-heavy and still not product-clean
- the replay run does not create a clear visible qualitative jump on this reference image

## Assessment
- This is a real new Stage-4 Lite2 run, not just a data-prep checkpoint.
- The replay-augmented model is not a promotion over the current Stage-3 baseline.
- What replay appears to help:
  - slightly lower FP load at low score thresholds
  - consistent precision gains for `person`
  - some BA-core improvement on `obstacle_fence` and `obstacle_pole`
- What replay does not convincingly help:
  - global AP / AP50 / AP75
  - recall on the common Stage-3 test split
  - broad stabilization of the six rehearsal classes
- Current verdict:
  - Stage-4 with the current small COCO replay is slightly worse than Stage-3 overall
  - keep Stage-3 as the preferred multi-source baseline on current repo HEAD
  - treat this Stage-4 result as useful negative evidence, not as a new default

## Next Step Boundary
- Do not widen BA-v1.
- Do not promote the current Stage-4 replay run into the main baseline.
- If replay is revisited, the next step should be a small weighting or selection adjustment inside the same six-class replay scope, not a broad new tuning project.
