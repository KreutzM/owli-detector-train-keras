# Dataset Runbook (PowerShell)

## Validate COCO

```powershell
python -m owli_train dataset validate --coco tests\data\coco_min.json
```

Optional image-file existence check:

```powershell
python -m owli_train dataset validate --coco data\instances.json --images-dir data\images
```

## Normalize COCO

Normalizes category IDs to deterministic contiguous IDs, optionally merging labels via label map.

```powershell
python -m owli_train dataset normalize --coco data\instances.json --out work\normalized\instances.json
```

With image checks and label mapping:

```powershell
python -m owli_train dataset normalize --coco data\instances.json --images-dir data\images --label-map configs\label_map.yaml --out work\normalized\instances.json
```

## Split COCO (train/val/test)

Writes `splits.json` deterministically by seed:

```powershell
python -m owli_train dataset split --coco data\instances.json --out-dir work\splits --seed 1337
```

Also emit per-split COCO files:

```powershell
python -m owli_train dataset split --coco data\instances.json --out-dir work\splits --seed 1337 --write-coco
```

Produces:
- `work\splits\splits.json`
- `work\splits\instances_train.json`
- `work\splits\instances_val.json`
- `work\splits\instances_test.json`

## Label map format

`configs\label_map.yaml` can be one of these forms:

```yaml
map:
  human: person
  automobile: car
```

```yaml
human: person
automobile: car
```

## Train detector (KerasCV YOLOv8 baseline)

Install training dependencies first:

```powershell
pip install -r requirements\keras.txt
```

Smoke run (bounded runtime):

```powershell
python -m owli_train train detect --config configs\train_detector.yaml --max-steps 1 --limit-train-images 8 --limit-val-images 4
```

Tiny local smoke dataset run:

```powershell
python -m owli_train train detect --config configs\train_detector_smoke.yaml --max-steps 1
```

Resume from a checkpoint:

```powershell
python -m owli_train train detect --config configs\train_detector.yaml --resume work\runs\<run_id>\checkpoints\epoch-001.weights.h5
```

Run artifacts:
- `work\runs\<run_id>\config.yaml`
- `work\runs\<run_id>\label_map_snapshot.json`
- `work\runs\<run_id>\logs\train.csv`
- `work\runs\<run_id>\checkpoints\*.weights.h5`
- `work\runs\<run_id>\artifacts\detector.keras`
- `work\runs\<run_id>\artifacts\saved_model\`

## Evaluate detector (COCO mAP)

Install evaluation deps:

```powershell
pip install -r requirements\eval.txt
```

Evaluate from a run directory:

```powershell
python -m owli_train eval detect --coco data\coco\instances_val.json --images-dir data\coco\images --run-dir work\runs\<run_id>
```

Evaluate from a direct model path:

```powershell
python -m owli_train eval detect --coco data\coco\instances_val.json --images-dir data\coco\images --model work\runs\<run_id>\artifacts\detector.keras --out work\reports\eval.json
```

Quick smoke eval:

```powershell
python -m owli_train eval detect --coco tests\data\coco_min.json --images-dir tests\data --run-dir work\runs\<run_id> --limit-images 1 --max-detections-per-image 10
```

Optional explicit class/category mapping:

```powershell
python -m owli_train eval detect --coco data\coco\instances_val.json --images-dir data\coco\images --run-dir work\runs\<run_id> --category-map configs\eval_category_map.json
```

Reports are written to:
- `work\runs\<run_id>\reports\eval.json`
- `work\runs\<run_id>\reports\eval.md`

## Export detector to TFLite

Default export from run artifacts (prefers `saved_model`, fallback `.keras`):

```powershell
python -m owli_train export tflite --run-dir work\runs\<run_id>
```

FP16 export:

```powershell
python -m owli_train export tflite --run-dir work\runs\<run_id> --quant fp16
```

Fail export when output requires Select TF Ops (Flex):

```powershell
python -m owli_train export tflite --run-dir work\runs\<run_id> --quant fp16 --require-builtins-only
```

INT8 export (with representative dataset):

```powershell
python -m owli_train export tflite --run-dir work\runs\<run_id> --quant int8 --rep-coco data\coco\instances_train.json --rep-images-dir data\coco\images --rep-max-images 32
```

Direct model export:

```powershell
python -m owli_train export tflite --model work\runs\<run_id>\artifacts\detector.keras --out outputs\detector.tflite
```

Export artifacts:
- `work\runs\<run_id>\artifacts\detector.tflite` (default for `--run-dir`)
- `...\detector.tflite.meta.json` (labels/class_names, bbox_format, input_size, settings, android_compat)

Inspect TFLite Android compatibility:

```powershell
python -m owli_train inspect tflite --model work\runs\<run_id>\artifacts\detector.tflite
```

For Builtins vs Flex details, see `docs\android-deploy.md`.

## Bench TFLite model

Bench from run-dir exported model:

```powershell
python -m owli_train bench tflite --run-dir work\runs\<run_id> --limit-images 8 --warmup-runs 3 --runs 16
```

Bench direct model path:

```powershell
python -m owli_train bench tflite --model work\runs\<run_id>\artifacts\detector.tflite --limit-images 2
```

Optional real-image bench input:

```powershell
python -m owli_train bench tflite --model work\runs\<run_id>\artifacts\detector.tflite --images-dir tests\smoke_coco\images --limit-images 2
```

Bench report path:
- `work\runs\<run_id>\reports\bench_tflite.json` (default with `--run-dir`)
- `work\reports\bench_tflite.json` (default with direct `--model`)
