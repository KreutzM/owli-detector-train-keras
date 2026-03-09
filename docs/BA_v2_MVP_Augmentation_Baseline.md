# BA-v2 MVP Augmentation Baseline

## Scope
- First real BA-v2 MVP comparison run with the now-verified online augmentation hook in the existing `EfficientDet-Lite2` / Model Maker path.
- Goal:
  - compare the current BA-v2 MVP baseline without augmentation against the smallest controlled online-augmentation variant
  - keep the run fair by changing only the train-time augmentation knobs

Reference baseline:
- [BA_v2_MVP_Baseline.md](./BA_v2_MVP_Baseline.md)

## Compared Configs
- Baseline:
  - [`configs/efficientdet_lite2_ba_v2_mvp.yaml`](../configs/efficientdet_lite2_ba_v2_mvp.yaml)
- Augmentation comparison:
  - [`configs/efficientdet_lite2_ba_v2_mvp_aug.yaml`](../configs/efficientdet_lite2_ba_v2_mvp_aug.yaml)

Only intended train-path difference:
- `train.augmentation.rand_hflip: true`
- `train.augmentation.jitter_min: 0.9`
- `train.augmentation.jitter_max: 1.1`

Deliberately unchanged:
- same BA-v2 MVP dataset and splits
- same `EfficientDet-Lite2` variant
- same `20` epochs
- same batch size `16`
- same `train_whole_model: false`
- no `autoaugment_policy`

## Run Identity
- Run dir:
  - `work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309`
- TFLite model:
  - `work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/artifacts/model.tflite`
- Model size:
  - `7.4 MB`
- Real train-to-export wall time:
  - about `31.3` minutes on the local WSL2 / RTX 3060 setup

## Verified Commands
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet \
  --config configs/efficientdet_lite2_ba_v2_mvp_aug.yaml \
  --run-name ba-v2-mvp-augmentation-baseline-20260309 \
  --require-gpu

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite \
  --model work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/artifacts/model.tflite

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_v2_hazard_slice02_mapillary_od_ground/instances_test.json \
  --images-dir work/datasets/ba_v2_mvp_candidate/images \
  --model work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 8 \
  --out work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/reports/eval_efficientdet_tflite_ba_v2_test.json

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect \
  --model work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/artifacts/model.tflite \
  --image work/datasets/ba_v2_mvp_candidate/images/mapillary_vistas/training/ppvi1a8kNPmFjkS6Lhbnsg.jpg \
  --out work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/reports/golden_ba_v2_test_mix.json \
  --score-threshold 0.1 \
  --max-results 20 \
  --num-threads 8
```

## Primary BA-v2 TEST Eval
- report JSON:
  - `work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/reports/eval_efficientdet_tflite_ba_v2_test.json`
- report Markdown:
  - `work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/reports/eval_efficientdet_tflite_ba_v2_test.md`
- eval images: `381`
- score threshold: `0.1`
- noise thresholds: `0.05`, `0.1`, `0.3`

Global metrics vs. the current BA-v2 MVP baseline:

| Variant | AP | AP50 | AP75 | AR100 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BA-v2 baseline` | `0.1184` | `0.2149` | `0.1120` | `0.2005` | `0.1948` | `0.3653` |
| `BA-v2 + online aug` | `0.1140` | `0.2107` | `0.1085` | `0.1948` | `0.1815` | `0.3434` |
| delta | `-0.0044` | `-0.0042` | `-0.0035` | `-0.0057` | `-0.0133` | `-0.0219` |

Aggregate counts:
- baseline: TP `1101`, FP `4551`, FN `1913`
- augmentation: TP `1035`, FP `4667`, FN `1979`

False-positive load:
- threshold `0.05`:
  - baseline FP/100 `2150.13`
  - augmentation FP/100 `2204.20`
  - delta `+54.07`
- threshold `0.10`:
  - baseline FP/100 `1194.49`
  - augmentation FP/100 `1224.93`
  - delta `+30.45`
- threshold `0.30`:
  - baseline FP/100 `105.77`
  - augmentation FP/100 `87.40`
  - delta `-18.37`

Reading:
- the first small online-augmentation run is globally slightly worse than the non-augmented BA-v2 baseline
- low-threshold FP pressure increases, while only the stricter `0.30` threshold improves
- this is not a collapse, but it is also not enough evidence to promote the augmented run as the new default

## Per-Class Comparison
Hazard-core classes:
- `obstacle_ground`:
  - precision `0.0313 -> 0.0239`
  - recall `0.3400 -> 0.3000`
  - FP `527 -> 612`
- `obstacle_barrier`:
  - precision `0.0712 -> 0.0680`
  - recall `0.3679 -> 0.4057`
  - TP `39 -> 43`
  - FP `509 -> 589`
- `obstacle_hole_dropoff`:
  - precision `0.0526 -> 0.0575`
  - recall `0.2985 -> 0.3134`
  - TP `20 -> 21`
  - FP `360 -> 344`
- `obstacle_pole`:
  - precision `0.1710 -> 0.1483`
  - recall `0.1536 -> 0.1536`
  - FP `674 -> 798`

Hazard-core reading:
- `obstacle_barrier` gains some recall, but pays for it with more false positives
- `obstacle_hole_dropoff` is the cleanest small winner in this run
- `obstacle_ground` gets worse
- `obstacle_pole` does not improve and becomes noisier

Rehearsal classes:
- `person`:
  - precision `0.2347 -> 0.2359`
  - recall `0.4767 -> 0.4630`
- `bicycle`:
  - precision `0.2126 -> 0.2093`
  - recall `0.2372 -> 0.2308`
- `motorcycle`:
  - precision `0.2193 -> 0.2540`
  - recall `0.3677 -> 0.2870`
- `car`:
  - precision `0.2957 -> 0.2880`
  - recall `0.5834 -> 0.5292`
- `bus`:
  - precision `0.3395 -> 0.2647`
  - recall `0.4044 -> 0.3971`
- `truck`:
  - precision `0.2327 -> 0.2333`
  - recall `0.3083 -> 0.2917`

Rehearsal reading:
- `motorcycle` reduces false positives strongly, but loses too much recall
- `car` loses the most relevant rehearsal recall
- `bus` gets noticeably noisier
- overall rehearsal behavior is slightly worse than the current baseline

## Golden Detect
- report:
  - `work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/reports/golden_ba_v2_test_mix.json`
- detections written: `20`
- class mix:
  - `person`: `10`
  - `car`: `8`
  - `truck`: `2`

Reading:
- as in the non-augmented BA-v2 baseline, the sample still collapses toward rehearsal classes
- the top-20 output still contains no hazard-core detections for this mixed sample
- the first small online-augmentation run does not change the qualitative product story yet

## Assessment
- The small online augmentation hook works technically in the existing BA-v2 MVP Model-Maker path.
- For this first real comparison run, the result is not strong enough to replace the current BA-v2 baseline.
- Honest current verdict:
  - keep the non-augmented BA-v2 MVP run as the preferred baseline
  - keep the online-augmentation hook available for controlled follow-up runs
  - treat `obstacle_hole_dropoff` and, with caution, `obstacle_barrier` as the only visible positive signal from this first comparison

## Recommended Next Step
- If online augmentation stays in scope, run exactly one narrower follow-up comparison with a weaker setting such as `rand_hflip` only or reduced jitter, instead of widening augmentation complexity.
