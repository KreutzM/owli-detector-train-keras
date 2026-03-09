# BA MVP Stage-3 Baseline

## Scope
- First real `EfficientDet-Lite2` / Model Maker baseline on the balanced multi-source BA-v1 dataset.
- Sources included:
  - `Obstacle4`
  - balanced `Mapillary Vistas`
  - `OD / Obstacle-Dataset`
- `COCO replay` is intentionally not part of this run.

## Run Identity
- Config: [`configs/efficientdet_lite2_ba_mvp_stage3.yaml`](../configs/efficientdet_lite2_ba_mvp_stage3.yaml)
- Canonical label contract for training: [`configs/label_contracts/ba_v1.class_names.json`](../configs/label_contracts/ba_v1.class_names.json)
- Run dir:
  - `work/runs/20260308-183140-ba-mvp-stage3-20260308`
- TFLite model:
  - `work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite`
- Model size:
  - `7.1 MB`

## Data Input
- Materialized multi-source COCO:
  - `work/datasets/ba_mvp_stage3_balanced_multisource/instances_materialized.json`
- Materialized images:
  - `work/datasets/ba_mvp_stage3_balanced_multisource/images`
- Model Maker CSV:
  - `work/datasets/ba_mvp_stage3_balanced_multisource/modelmaker.csv`
- Split source:
  - `work/splits/ba_mvp_stage3_balanced_multisource/splits.json`

Source mix in the merged Stage-3 dataset:
- `Obstacle4`: `1250` images, `1912` annotations
- balanced `Mapillary`: `1224` images, `27578` annotations
- `OD`: `1592` images, `8909` annotations
- total: `4066` images, `38399` annotations, `10` categories

Held-out `TEST` split used for the primary Stage-3 eval:
- images: `408`
- annotations: `3874`

## Verified Commands
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet \
  --config configs/efficientdet_lite2_ba_mvp_stage3.yaml \
  --run-name ba-mvp-stage3-20260308 \
  --require-gpu

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite

PYTHONPATH=src python -m owli_train dataset split \
  --coco work/datasets/ba_mvp_stage3_balanced_multisource/instances_combined.json \
  --out-dir work/splits/ba_mvp_stage3_balanced_multisource \
  --seed 1337 \
  --ensure-train-class-coverage \
  --write-coco

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json \
  --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 8 \
  --out work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_stage3_test.json

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/datasets/obstacle4/instances_combined.json \
  --images-dir data/raw/obstacle4/extracted \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 16 \
  --out work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_obstacle4.json

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite \
  --image data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg \
  --out work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/golden_obstacle4.json \
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

Current BA-v1 naming remains `obstacle_fence` and `obstacle_hole`. The older names
`obstacle_fence_rail` and `obstacle_hole_dropoff` are not part of the current repo contract.

## Training Result
- completed epochs: `20`
- final reported losses:
  - `val_det_loss`: `0.5496`
  - `val_cls_loss`: `0.4129`
  - `val_box_loss`: `0.0027`
  - `val_loss`: `0.5569`

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

## Primary Stage-3 Eval
Primary quantitative readout uses the held-out Stage-3 `TEST` split:
- report JSON:
  - `work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_stage3_test.json`
- report Markdown:
  - `work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_stage3_test.md`
- eval images: `408`
- score threshold: `0.1`
- noise thresholds: `0.05`, `0.1`, `0.3`

Global metrics:
- AP@[0.50:0.95]: `0.1307`
- AP50: `0.2325`
- AP75: `0.1270`
- AR100: `0.2170`
- TP: `1447`
- FP: `5612`
- FN: `2427`
- Precision: `0.2050`
- Recall: `0.3735`

Noise:
- threshold `0.05`: FP `8601`, FP/100 `2108.09`
- threshold `0.10`: FP `5612`, FP/100 `1375.49`
- threshold `0.30`: FP `455`, FP/100 `111.52`

Per-class aggregate on the Stage-3 `TEST` split:
- `obstacle_bump`: TP `22`, FP `524`, FN `20`, precision `0.0403`, recall `0.5238`
- `obstacle_fence`: TP `23`, FP `350`, FN `92`, precision `0.0617`, recall `0.2000`
- `obstacle_hole`: TP `19`, FP `495`, FN `30`, precision `0.0370`, recall `0.3878`
- `obstacle_pole`: TP `163`, FP `962`, FN `964`, precision `0.1449`, recall `0.1446`
- `bicycle`: TP `78`, FP `227`, FN `164`, precision `0.2557`, recall `0.3223`
- `bus`: TP `66`, FP `151`, FN `104`, precision `0.3041`, recall `0.3882`
- `car`: TP `553`, FP `1245`, FN `417`, precision `0.3076`, recall `0.5701`
- `motorcycle`: TP `130`, FP `337`, FN `192`, precision `0.2784`, recall `0.4037`
- `person`: TP `344`, FP `1149`, FN `351`, precision `0.2304`, recall `0.4950`
- `truck`: TP `49`, FP `172`, FN `93`, precision `0.2217`, recall `0.3451`

Interpretation:
- all `10` BA-v1 classes now produce non-zero true positives on the held-out Stage-3 test split
- rehearsal classes are no longer effectively dead
- `car`, `person`, `motorcycle`, `bus`, and `bicycle` clearly benefit from the multi-source dataset
- obstacle precision remains weak, especially `obstacle_bump`, `obstacle_fence`, and `obstacle_hole`
- `obstacle_pole` remains difficult despite abundant data, which points to heavy ambiguity / background clutter rather than only data scarcity

## Obstacle4 Comparison Run
For continuity with the existing reference baseline, the new Stage-3 model was also evaluated on the full `Obstacle4` combined dataset:
- report JSON:
  - `work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_obstacle4.json`
- report Markdown:
  - `work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_obstacle4.md`

Global result on `Obstacle4` full (`1250` images):
- AP@[0.50:0.95]: `0.2443`
- AP50: `0.3827`
- AP75: `0.2548`
- AR100: `0.3926`
- TP: `1118`
- FP: `12475`
- FN: `794`
- Precision: `0.0822`
- Recall: `0.5847`

Direct comparison to the old Obstacle4-only baseline from [Obstacle4_E2E_Results.md](./Obstacle4_E2E_Results.md):
- AP improved from `0.0952` to `0.2443`
- AP50 improved from `0.1897` to `0.3827`
- AP75 improved from `0.0886` to `0.2548`
- AR100 improved from `0.1899` to `0.3926`
- aggregate precision changed from `0.0881` to `0.0822`
- aggregate recall changed from `0.6219` to `0.5847`
- FP/100 at `0.10` changed from `984.8` to `998.0`
- FP/100 at `0.30` improved from `51.2` to `39.2`

What improved visibly on the Obstacle4 comparison:
- `person`, `bicycle`, `motorcycle`, `bus`, and `truck` now produce real detections instead of staying at zero
- `car` remains strong and improves slightly
- `obstacle_pole` recall improves

What regressed or remains weak on the Obstacle4 comparison:
- `obstacle_fence` recall drops while precision improves
- `obstacle_bump` and `obstacle_hole` still carry heavy false-positive load
- aggregate FP load at low score threshold remains high

## Golden Detect
- report:
  - `work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/golden_obstacle4.json`
- input image:
  - `data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg`
- detections written: `20`
- class mix:
  - `obstacle_pole`: `8`
  - `car`: `6`
  - `obstacle_hole`: `4`
  - `obstacle_fence`: `2`

Reading:
- the sample is no longer mostly `obstacle_pole` plus stray `car`
- the model surfaces more class variety, but the output is still over-dense and not yet product-clean

## Assessment
- This is a real new Stage-3 baseline, not just a data-prep checkpoint.
- The balanced multi-source dataset clearly improves the detector relative to the old Obstacle4-only reference.
- The biggest gain is not obstacle precision but label coverage and non-zero signal on the previously dead rehearsal classes.
- The model is still not product-ready:
  - false-positive pressure remains high
  - obstacle classes are noisy
  - the next step should focus on stabilizing rehearsal classes and reducing FP load, not on widening the label contract

## Next Step Boundary
- `COCO replay` is still intentionally absent here.
- The current product-near operating-point check for this baseline is documented in
  [BA_MVP_Stage3_Product_Gate.md](./BA_MVP_Stage3_Product_Gate.md).
- The next obstacle-focused experiment branch is now the small-object crop path in
  [BA_MVP_Stage3_Crops.md](./BA_MVP_Stage3_Crops.md).
- The first direct `Stage-3` vs. `Stage-3-plus-crops` result is now documented in
  [BA_MVP_Stage3_Plus_Crops_Baseline.md](./BA_MVP_Stage3_Plus_Crops_Baseline.md).
- That comparison does not replace the current full-image Stage-3 setup; the preferred
  multi-source baseline remains Stage-3.
- The next baseline step should add a small explicit replay subset for:
  - `person`
  - `bicycle`
  - `motorcycle`
  - `car`
  - `bus`
  - `truck`
