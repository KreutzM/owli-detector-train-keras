# BA MVP Stage-3-plus-crops Comparison Run

## Scope
- First real `EfficientDet-Lite2` / Model Maker run on the prepared `Stage-3-plus-crops` dataset.
- Direct comparison target stays the verified `Stage-3` baseline from [BA_MVP_Stage3_Baseline.md](./BA_MVP_Stage3_Baseline.md).
- This run evaluates whether the added small-object crop branch is enough to beat or replace the current Stage-3 baseline.

## Run Identity
- Config: [`configs/efficientdet_lite2_ba_mvp_stage3_plus_crops.yaml`](../configs/efficientdet_lite2_ba_mvp_stage3_plus_crops.yaml)
- Canonical label contract for training: [`configs/label_contracts/ba_v1.class_names.json`](../configs/label_contracts/ba_v1.class_names.json)
- Run dir:
  - `work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309`
- TFLite model:
  - `work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/artifacts/model.tflite`
- Model size:
  - `7.1 MB`
- Real train-to-export wall time:
  - about `35.6` minutes on the local WSL2 / RTX 3060 setup

## Data Input
- Materialized combined COCO:
  - `work/datasets/ba_mvp_stage3_plus_crops/instances_materialized.json`
- Materialized images:
  - `work/datasets/ba_mvp_stage3_plus_crops/images`
- Model Maker CSV:
  - `work/datasets/ba_mvp_stage3_plus_crops/modelmaker.csv`
- Comparison eval set:
  - held-out `TEST` split from `work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json`

Prepared dataset size:
- `4594` images
- `41400` annotations
- `10` categories

Fairness notes:
- same `EfficientDet-Lite2` variant as Stage-3
- same `20` epochs
- same batch size `16`
- same BA-v1 label order via [`configs/label_contracts/ba_v1.class_names.json`](../configs/label_contracts/ba_v1.class_names.json)
- same held-out Stage-3 `TEST` split for the direct comparison
- practical difference is only the additional crop rows in `TRAIN`

## Verified Commands
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet \
  --config configs/efficientdet_lite2_ba_mvp_stage3_plus_crops.yaml \
  --run-name ba-mvp-stage3-plus-crops-20260309 \
  --require-gpu

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite \
  --model work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/artifacts/model.tflite

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json \
  --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --model work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 8 \
  --out work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/reports/eval_efficientdet_tflite_stage3_test.json

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect \
  --model work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/artifacts/model.tflite \
  --image data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg \
  --out work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/reports/golden_obstacle4.json \
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

## Primary Eval On The Stage-3 TEST Split
- report JSON:
  - `work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/reports/eval_efficientdet_tflite_stage3_test.json`
- report Markdown:
  - `work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/reports/eval_efficientdet_tflite_stage3_test.md`
- eval images: `408`
- score threshold: `0.1`
- noise thresholds: `0.05`, `0.1`, `0.3`

Global metrics:
- AP@[0.50:0.95]: `0.1280`
- AP50: `0.2276`
- AP75: `0.1202`
- AR100: `0.2142`
- TP: `1427`
- FP: `5424`
- FN: `2447`
- Precision: `0.2083`
- Recall: `0.3684`

Noise:
- threshold `0.05`: FP `8493`, FP/100 `2081.62`
- threshold `0.10`: FP `5424`, FP/100 `1329.41`
- threshold `0.30`: FP `478`, FP/100 `117.16`

Per-class aggregate on the Stage-3 `TEST` split:
- `obstacle_bump`: TP `16`, FP `415`, FN `26`, precision `0.0371`, recall `0.3810`
- `obstacle_fence`: TP `29`, FP `308`, FN `86`, precision `0.0861`, recall `0.2522`
- `obstacle_hole`: TP `17`, FP `356`, FN `32`, precision `0.0456`, recall `0.3469`
- `obstacle_pole`: TP `172`, FP `1177`, FN `955`, precision `0.1275`, recall `0.1526`
- `bicycle`: TP `79`, FP `226`, FN `163`, precision `0.2590`, recall `0.3264`
- `bus`: TP `66`, FP `172`, FN `104`, precision `0.2773`, recall `0.3882`
- `car`: TP `539`, FP `1210`, FN `431`, precision `0.3082`, recall `0.5557`
- `motorcycle`: TP `135`, FP `350`, FN `187`, precision `0.2784`, recall `0.4193`
- `person`: TP `330`, FP `1066`, FN `365`, precision `0.2364`, recall `0.4748`
- `truck`: TP `44`, FP `144`, FN `98`, precision `0.2340`, recall `0.3099`

## Direct Comparison Vs. The Verified Stage-3 Baseline
On the same held-out Stage-3 `TEST` split:
- AP drops from `0.1307` to `0.1280`
- AP50 drops from `0.2325` to `0.2276`
- AP75 drops from `0.1270` to `0.1202`
- AR100 drops from `0.2170` to `0.2142`
- precision improves slightly from `0.2050` to `0.2083`
- recall drops from `0.3735` to `0.3684`
- FP at threshold `0.10` improves from `5612` to `5424`
- FP/100 at threshold `0.10` improves from `1375.49` to `1329.41`
- FP/100 at threshold `0.30` regresses from `111.52` to `117.16`

BA-core reading:
- `obstacle_fence` is the clearest winner:
  - TP `23 -> 29`
  - precision `0.0617 -> 0.0861`
  - recall `0.2000 -> 0.2522`
- `obstacle_pole` gains a little recall:
  - TP `163 -> 172`
  - recall `0.1446 -> 0.1526`
  - but FP rises `962 -> 1177`, so precision drops
- `obstacle_hole` lowers FP materially:
  - FP `495 -> 356`
  - precision improves `0.0370 -> 0.0456`
  - but recall drops `0.3878 -> 0.3469`
- `obstacle_bump` regresses:
  - TP `22 -> 16`
  - recall `0.5238 -> 0.3810`

Rehearsal-class reading:
- small positive movement:
  - `bicycle` improves slightly on both precision and recall
  - `motorcycle` gains recall with nearly unchanged precision
- mixed or negative movement:
  - `person` precision improves a little, recall drops
  - `car` stays roughly flat on precision, recall drops
  - `bus` recall is flat but precision drops
  - `truck` precision improves but recall drops

Extra metric nuance:
- `AP_small` moves only slightly upward from `0.0160` to `0.0165`
- `AP_medium` improves from `0.0745` to `0.0773`
- those gains are too small to offset the global drop on the full comparison set

## Golden Detect
- report:
  - `work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/reports/golden_obstacle4.json`
- input image:
  - `data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg`
- detections written: `15`
- class mix:
  - `obstacle_pole`: `7`
  - `car`: `3`
  - `obstacle_hole`: `3`
  - `obstacle_fence`: `1`
  - `obstacle_bump`: `1`

Reading:
- the sample is slightly less over-dense than the Stage-3 baseline (`15` vs. `20` detections)
- crop training does surface one explicit `obstacle_bump` prediction on the sample
- the output is still dominated by `obstacle_pole` and is not evidence of a product-clean improvement by itself

## Assessment
- This is a real completed Stage-3-plus-crops comparison run, not just a prepared dataset.
- The extra crop signal is not strong enough to replace the current Stage-3 baseline.
- Most honest reading:
  - useful local signal for `obstacle_fence`
  - maybe a recall nudge for `obstacle_pole`
  - lower low-threshold FP load overall
  - but not enough to hold or improve the main global metrics
- `obstacle_bump` remains badly under-served even with the first crop branch.
- `obstacle_hole` stays mixed: lower FP, lower recall.
- The current preferred multi-source baseline should remain `Stage-3`.

## Next Step Boundary
- Keep the current `Stage-3` full-image baseline as the preferred comparison anchor.
- Treat the crop branch as an explored experiment, not as the new default baseline.
- If the crop idea is revisited, the next small step should tighten the crop branch toward:
  - more useful `obstacle_bump` coverage
  - less `obstacle_pole` FP amplification
  - no change to the BA-v1 contract
