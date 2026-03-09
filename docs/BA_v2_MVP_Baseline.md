# BA-v2 MVP Baseline

## Scope
- First real `EfficientDet-Lite2` / Model Maker baseline on the prepared BA-v2 MVP dataset.
- Goal:
  - establish the first honest BA-v2 hazard-centered baseline on current repo HEAD
  - keep the run small and reviewable rather than tuned for a final optimum
- Follow-up comparison:
  - [BA_v2_MVP_Augmentation_Baseline.md](./BA_v2_MVP_Augmentation_Baseline.md)
  - result: the first small online-augmentation comparison did not beat this baseline and does not replace it as the preferred BA-v2 MVP reference

## Run Identity
- Config: [`configs/efficientdet_lite2_ba_v2_mvp.yaml`](../configs/efficientdet_lite2_ba_v2_mvp.yaml)
- Canonical label contract for training: [`configs/label_contracts/ba_v2_hazard.class_names.json`](../configs/label_contracts/ba_v2_hazard.class_names.json)
- Run dir:
  - `work/runs/20260309-111756-ba-v2-mvp-baseline-20260309`
- TFLite model:
  - `work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/artifacts/model.tflite`
- Model size:
  - `7.4 MB`
- Real train-to-export wall time:
  - about `30.6` minutes on the local WSL2 / RTX 3060 setup

## Data Input
- Materialized BA-v2 MVP COCO:
  - `work/datasets/ba_v2_mvp_candidate/instances_materialized.json`
- Materialized images:
  - `work/datasets/ba_v2_mvp_candidate/images`
- Model Maker CSV:
  - `work/datasets/ba_v2_mvp_candidate/modelmaker.csv`
- Split source:
  - `work/splits/ba_v2_hazard_slice02_mapillary_od_ground/splits.json`

Prepared dataset size:
- `3799` images
- `32231` annotations
- `10` categories

Held-out BA-v2 `TEST` split used for the primary baseline eval:
- images: `381`
- annotations: `3014`

Fairness notes:
- same `EfficientDet-Lite2` variant as the historical Stage-3 / Stage-4 / Stage-3-plus-crops runs
- same `20` epochs
- same batch size `16`
- same Model Maker export path
- different ontology and different held-out dataset than the BA-v1 historical baselines, so cross-baseline numbers are informative but not apples-to-apples

## Verified Commands
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet \
  --config configs/efficientdet_lite2_ba_v2_mvp.yaml \
  --run-name ba-v2-mvp-baseline-20260309 \
  --require-gpu

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite \
  --model work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/artifacts/model.tflite

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_v2_hazard_slice02_mapillary_od_ground/instances_test.json \
  --images-dir work/datasets/ba_v2_mvp_candidate/images \
  --model work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 8 \
  --out work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/reports/eval_efficientdet_tflite_ba_v2_test.json

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect \
  --model work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/artifacts/model.tflite \
  --image work/datasets/ba_v2_mvp_candidate/images/mapillary_vistas/training/ppvi1a8kNPmFjkS6Lhbnsg.jpg \
  --out work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/reports/golden_ba_v2_test_mix.json \
  --score-threshold 0.1 \
  --max-results 20 \
  --num-threads 8
```

## Final Label Contract
The exported Lite2 artifacts preserve the canonical BA-v2 MVP order:
1. `obstacle_ground`
2. `obstacle_barrier`
3. `obstacle_hole_dropoff`
4. `obstacle_pole`
5. `person`
6. `bicycle`
7. `motorcycle`
8. `car`
9. `bus`
10. `truck`

Alignment checks:
- `labels.txt` and `class_names.json` match the BA-v2 hazard order exactly
- `mapping_files.json` reports no missing classes from `TRAIN`
- TFLite eval aligned categories by label-name match from exported `labels.txt`

## Training Result
- completed epochs: `20`
- final reported train losses:
  - `det_loss`: `1.0414`
  - `cls_loss`: `0.6479`
  - `box_loss`: `0.0079`
  - `loss`: `1.0487`
- final reported validation losses:
  - `val_det_loss`: `1.2062`
  - `val_cls_loss`: `0.7814`
  - `val_box_loss`: `0.0085`
  - `val_loss`: `1.2136`

Reading:
- the run completed cleanly and exported the expected Lite2 artifacts
- the validation loss remains materially higher than the stronger BA-v1 historical runs, which already points to a harder or less mature BA-v2 hazard learning problem

## TFLite Inspect
- builtin ops only: `true`
- input preprocessing:
  - shape `1 x 448 x 448 x 3`
  - dtype `uint8`
  - resize policy `letterbox_square`
  - color space `RGB`
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

## Primary BA-v2 TEST Eval
- report JSON:
  - `work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/reports/eval_efficientdet_tflite_ba_v2_test.json`
- report Markdown:
  - `work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/reports/eval_efficientdet_tflite_ba_v2_test.md`
- eval images: `381`
- score threshold: `0.1`
- noise thresholds: `0.05`, `0.1`, `0.3`

Global metrics:
- AP@[0.50:0.95]: `0.1184`
- AP50: `0.2149`
- AP75: `0.1120`
- AR100: `0.2005`
- TP: `1101`
- FP: `4551`
- FN: `1913`
- Precision: `0.1948`
- Recall: `0.3653`

Noise:
- threshold `0.05`: FP `8192`, FP/100 `2150.13`
- threshold `0.10`: FP `4551`, FP/100 `1194.49`
- threshold `0.30`: FP `403`, FP/100 `105.77`

Per-class aggregate on the BA-v2 `TEST` split:
- `obstacle_ground`: TP `17`, FP `527`, FN `33`, precision `0.0313`, recall `0.3400`
- `obstacle_barrier`: TP `39`, FP `509`, FN `67`, precision `0.0712`, recall `0.3679`
- `obstacle_hole_dropoff`: TP `20`, FP `360`, FN `47`, precision `0.0526`, recall `0.2985`
- `obstacle_pole`: TP `139`, FP `674`, FN `766`, precision `0.1710`, recall `0.1536`
- `person`: TP `245`, FP `799`, FN `269`, precision `0.2347`, recall `0.4767`
- `bicycle`: TP `37`, FP `137`, FN `119`, precision `0.2126`, recall `0.2372`
- `motorcycle`: TP `82`, FP `292`, FN `141`, precision `0.2193`, recall `0.3677`
- `car`: TP `430`, FP `1024`, FN `307`, precision `0.2957`, recall `0.5834`
- `bus`: TP `55`, FP `107`, FN `81`, precision `0.3395`, recall `0.4044`
- `truck`: TP `37`, FP `122`, FN `83`, precision `0.2327`, recall `0.3083`

Class-level reading:
- all `10` BA-v2 MVP classes produce non-zero true positives on the held-out BA-v2 test split
- the clearest already-usable signal is still in the rehearsal side:
  - `car`
  - `person`
  - `bus`
- partial but still weak rehearsal signal exists for:
  - `motorcycle`
  - `truck`
  - `bicycle`
- hazard-core remains the main weakness:
  - `obstacle_ground` and `obstacle_hole_dropoff` are not usable yet
  - `obstacle_barrier` has some recall signal but still very weak precision
  - `obstacle_pole` stays difficult despite abundant data and still carries heavy ambiguity / clutter pressure

## Golden Detect
- report:
  - `work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/reports/golden_ba_v2_test_mix.json`
- input image:
  - `work/datasets/ba_v2_mvp_candidate/images/mapillary_vistas/training/ppvi1a8kNPmFjkS6Lhbnsg.jpg`
- chosen sample ground-truth class mix on the BA-v2 `TEST` split:
  - `bus`
  - `car`
  - `motorcycle`
  - `obstacle_barrier`
  - `obstacle_hole_dropoff`
  - `obstacle_pole`
  - `person`
  - `truck`
- detections written: `20`
- class mix:
  - `person`: `11`
  - `car`: `8`
  - `truck`: `1`

Reading:
- on a sample that contains three hazard-core classes, the top-20 exported detections are still dominated by rehearsal actors and vehicles
- the model does not surface any hazard-core detections in the written top-20 output for this sample
- this is consistent with the quantitative readout that BA-v2 obstacle semantics are still underpowered relative to the rehearsal side

## Assessment
- This is a real completed BA-v2 MVP Lite2 baseline, not a data-prep checkpoint.
- Product logic is clearly better than the historical BA-v1 obstacle-shaped contracts:
  - `obstacle_ground`
  - `obstacle_barrier`
  - `obstacle_hole_dropoff`
  - `obstacle_pole`
  read closer to the intended product behavior than `obstacle_bump`, `obstacle_fence`, and `obstacle_hole`.
- The result is still only first usable evidence, not a product-ready detector:
  - false-positive load remains high
  - the strongest hazard classes are still too noisy
  - the qualitative sample still collapses toward `person` / `car`
- Most honest current reading:
  - BA-v2 is the right ontology direction
  - the current BA-v2 data path is good enough to prove the contract is trainable end-to-end
  - the current BA-v2 model quality is not yet strong enough to claim a decisive promotion over the historical Stage-3 technical baseline

## Historical Comparison
Important boundary:
- the BA-v2 baseline above is evaluated on a different ontology and a different held-out dataset than the BA-v1 historical baselines below
- the comparison is therefore qualitative and product-facing, not a strict leaderboard

Historical reference metrics:

| Baseline | Eval set | AP | AP50 | AP75 | AR100 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Obstacle4-only` | full `Obstacle4` combined | `0.0952` | `0.1897` | `0.0886` | `0.1899` | `0.0881` | `0.6219` |
| `Stage-3 multi-source` | held-out Stage-3 `TEST` | `0.1307` | `0.2325` | `0.1270` | `0.2170` | `0.2050` | `0.3735` |
| `Stage-4 replay` | Stage-3 `TEST` comparison set | `0.1232` | `0.2170` | `0.1203` | `0.2095` | `0.2118` | `0.3627` |
| `Stage-3-plus-crops` | Stage-3 `TEST` comparison set | `0.1280` | `0.2276` | `0.1202` | `0.2142` | `0.2083` | `0.3684` |
| `BA-v2 MVP` | held-out BA-v2 `TEST` | `0.1184` | `0.2149` | `0.1120` | `0.2005` | `0.1948` | `0.3653` |

Qualitative readout:
- versus `Obstacle4-only`:
  - BA-v2 is much better product logic
  - rehearsal coverage is clearly healthier than the old single-source path
  - hazard precision is still not strong enough to call the BA-v2 detector good
- versus `Stage-3 multi-source`:
  - Stage-3 remains the stronger historical technical baseline on its own verified comparison set
  - BA-v2 has the cleaner ontology and is the preferred future product path
  - BA-v2 still needs more defended hazard-core data quality before it can replace Stage-3 as the strongest overall evidence
- versus `Stage-4 replay`:
  - BA-v2 avoids carrying forward the less-convincing replay tweak as the main story
  - rehearsal classes remain present without needing the replay branch
  - hazard-core quality is still not clearly stronger than the old BA-v1 multi-source path
- versus `Stage-3-plus-crops`:
  - BA-v2 is the more relevant product contract
  - the crop branch remains a historical BA-v1 experiment rather than a preferred direction
  - neither path solves the main false-positive problem yet

Current verdict:
- preferred product ontology path: `BA-v2 MVP`
- preferred currently strongest historical technical baseline for comparison: `Stage-3`
- preferred next real improvement step: stay inside the BA-v2 MVP contract and improve hazard-core data quality / false-positive control rather than widening the scope again
