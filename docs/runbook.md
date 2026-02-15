# Dataset Runbook (PowerShell + WSL)

## WSL quick setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements/dev.txt
```

WSL-specific setup notes: `docs/wsl-setup.md`

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

## Import YOLO dataset -> COCO

Download COCO128 (YOLO format):

```powershell
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip" -OutFile "data\coco128.zip"
Expand-Archive -Path "data\coco128.zip" -DestinationPath "data" -Force
```

WSL equivalent:

```bash
curl -L "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip" -o data/coco128.zip
unzip -oq data/coco128.zip -d data
```

Convert YOLO labels to COCO JSON:

```powershell
python -m owli_train dataset import yolo --yolo-dir data\coco128 --out work\datasets\coco128\instances.json
```

Outputs:
- `work\datasets\coco128\instances.json`
- `work\datasets\coco128\class_names.json`

## Export COCO -> Model Maker CSV

Export rows compatible with `object_detector.DataLoader.from_csv`:

```powershell
python -m owli_train dataset export modelmaker-csv --coco work\datasets\coco128\instances.json --images-dir data\coco128\images --out work\datasets\coco128\modelmaker.csv
```

Optional split mapping:

```powershell
python -m owli_train dataset export modelmaker-csv --coco work\datasets\coco128\instances.json --images-dir data\coco128\images --splits-json work\splits\splits.json --out work\datasets\coco128\modelmaker.csv
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

Override architecture at runtime:

```powershell
python -m owli_train train detect --config configs\train_detector.yaml --arch retinanet --max-steps 1
```

Tiny local smoke dataset run:

```powershell
python -m owli_train train detect --config configs\train_detector_smoke.yaml --max-steps 1
```

Builtins-first tiny smoke run (RetinaNet):

```powershell
python -m owli_train train detect --config configs\train_detector_builtins_smoke.yaml --max-steps 1
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

## Train EfficientDet-Lite (TFLite Model Maker)

Install Model Maker dependencies:

```powershell
pip install -r requirements\modelmaker.txt
```

Note: this backend is best run in a dedicated venv, separate from the KerasCV detector env.

WSL equivalent:

```bash
pip install -r requirements/modelmaker.txt
```

Train EfficientDet-Lite2 with config:

```powershell
python -m owli_train train efficientdet --config configs\efficientdet_lite2_coco128.yaml --max-steps 1
```

Require GPU (fail fast if TensorFlow cannot see a GPU in the active Model Maker interpreter):

```powershell
python -m owli_train train efficientdet --config configs\efficientdet_lite2_coco128.yaml --max-steps 500 --subset-seed 1337 --require-gpu
```

Use `--subset-seed` to make `--max-steps` subset selection deterministic:

```powershell
python -m owli_train train efficientdet --config configs\efficientdet_lite2_coco128.yaml --max-steps 5000 --subset-seed 1337
```

Override variant at runtime (`lite0..lite4`):

```powershell
python -m owli_train train efficientdet --config configs\efficientdet_lite2_coco128.yaml --variant lite3 --max-steps 1
```

Class-order invariant for pretrained Lite models:
- Keep `data.label_map_json` aligned to the canonical dataset class order.
- The training pipeline canonicalizes CSV class order from `label_map_json` before Model Maker load.
- If loaded class indices still mismatch, training aborts with a clear error to prevent corrupted fine-tuning.

Run artifacts:
- `work\runs\<run_id>\config.yaml`
- `work\runs\<run_id>\mapping_files.json`
- `work\runs\<run_id>\artifacts\model.tflite`
- `work\runs\<run_id>\artifacts\labels.txt`
- `work\runs\<run_id>\artifacts\class_names.json`

## Generate COCO-80 pseudo labels (GPU teacher: TF2 SavedModel / TF Hub)

Use a dedicated teacher venv so heavy TF Hub dependencies stay isolated.

PowerShell setup:

```powershell
python -m venv .venv-teacher
.\.venv-teacher\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements\teacher.txt
$env:TEACHER_PYTHON_EXE=".\.venv-teacher\Scripts\python.exe"
```

WSL setup:

```bash
python3 -m venv .venv-teacher
source .venv-teacher/bin/activate
python -m pip install --upgrade pip
pip install -r requirements/teacher.txt
export TEACHER_PYTHON_EXE=.venv-teacher/bin/python
```

Sanity check GPU visibility (teacher env):

```bash
$TEACHER_PYTHON_EXE -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Pseudo-label command (COCO-80 categories):

```powershell
python -m owli_train dataset pseudo-label coco --images-dir data\ba\images --out work\pseudo\ba_pseudo_coco80.json --limit-images 50 --batch-size 16 --score-threshold 0.6 --max-detections-per-image 50
```

WSL equivalent:

```bash
python -m owli_train dataset pseudo-label coco --images-dir data/ba/images --out work/pseudo/ba_pseudo_coco80.json --limit-images 50 --batch-size 16 --score-threshold 0.6 --max-detections-per-image 50
```

Outputs:
- pseudo COCO JSON: `--out`
- QC report JSON: defaults to `<out>.report.json` (override via `--report-out`)

Useful options:
- `--teacher <tfhub_handle>` or `--teacher-savedmodel <dir>`
- `--classes person,car` (name/id filter)
- `--num-parallel-calls`, `--prefetch-buffer`
- `--debug-io` (print teacher output signature + tensor shapes)
- `--viz-out-dir <dir>` (optional visual sample export; disabled by default)

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

## Evaluate EfficientDet TFLite (Model Maker export)

Install both dependency sets:

```powershell
pip install -r requirements\eval.txt
pip install -r requirements\modelmaker.txt
```

If Model Maker lives in a separate venv, set `MODELMAKER_PYTHON_EXE` so the command auto-delegates:

```powershell
$env:MODELMAKER_PYTHON_EXE=".\.venv-modelmaker-py39\Scripts\python.exe"
python -m owli_train eval efficientdet-tflite --coco work\datasets\coco128\instances_val.json --images-dir data\coco128\images\train2017 --model work\runs\<run_id>\artifacts\model.tflite --limit-images 20
```

WSL equivalent:

```bash
MODELMAKER_PYTHON_EXE=.venv-modelmaker-py39/bin/python python -m owli_train eval efficientdet-tflite --coco work/datasets/coco128/instances_val.json --images-dir data/coco128/images/train2017 --model work/runs/<run_id>/artifacts/model.tflite --limit-images 20
```

When all deps are already installed in the active environment, leave `MODELMAKER_PYTHON_EXE` unset and run directly:

```powershell
python -m owli_train eval efficientdet-tflite --coco data\coco\instances_val.json --images-dir data\coco\images --model work\runs\<run_id>\artifacts\model.tflite --out work\reports\eval_efficientdet_tflite.json
```

Optional explicit category mapping:

```powershell
python -m owli_train eval efficientdet-tflite --coco data\coco\instances_val.json --images-dir data\coco\images --model work\runs\<run_id>\artifacts\model.tflite --category-map configs\eval_category_map.yaml
```

Report content:
- COCO metrics (AP/AP50/AP75/AR*)
- Per-class TP/FP/FN plus precision/recall summary
- Noise metric: `FP per 100 images` at `--score-threshold` (default `0.3`)

Default output paths:
- `work\runs\<run_id>\reports\eval_efficientdet_tflite.json` + `.md` (when model is under `artifacts\`)
- `work\reports\eval-efficientdet-tflite-<timestamp>.json` + `.md` (direct model path)

### COCO val2017 compare (fine-tuned vs baseline)

Expected COCO layout:
- `data/coco2017/val2017/*.jpg`
- `data/coco2017/annotations/instances_val2017.json`

Bootstrap dataset + baseline model (PowerShell):

```powershell
.\scripts\fetch_coco2017_val.ps1 -CocoRoot data\coco2017 -WithBaseline -BaselineOut work\models\efficientdet_lite2_baseline.tflite
```

Bootstrap dataset + baseline model (WSL/bash):

```bash
bash scripts/fetch_coco2017_val.sh --coco-root data/coco2017 --with-baseline --baseline-out work/models/efficientdet_lite2_baseline.tflite
```

Run the compare wrapper (WSL/bash):

```bash
MODELMAKER_PYTHON_EXE=.venv-modelmaker-py39/bin/python \
bash scripts/eval_coco_val2017.sh \
  --coco-root data/coco2017 \
  --fine-tuned-model work/runs/<run_id>/artifacts/model.tflite \
  --baseline-model work/models/efficientdet_lite2_baseline.tflite \
  --limit-images 5000 \
  --max-detections 100 \
  --num-threads 8 \
  --noise-thresholds 0.05,0.1,0.3
```

Optional auto-bootstrap when files are missing:

```bash
MODELMAKER_PYTHON_EXE=.venv-modelmaker-py39/bin/python \
bash scripts/eval_coco_val2017.sh \
  --coco-root data/coco2017 \
  --fine-tuned-model work/runs/<run_id>/artifacts/model.tflite \
  --baseline-model work/models/efficientdet_lite2_baseline.tflite \
  --download-coco-if-missing \
  --download-baseline-if-missing
```

Outputs:
- Combined Markdown: `work/reports/val2017_compare/COCO2017_Val_Eval_Report.md`
- Combined JSON: `work/reports/val2017_compare/COCO2017_Val_Eval_Report.json`
- Raw per-model eval JSON:
  - `work/reports/val2017_compare/fine_tuned_eval.json`
  - `work/reports/val2017_compare/baseline_eval.json`

Optional committed docs snapshot:
- `--out-md docs/COCO2017_Val_Eval_Report.md`
- `--out-json docs/COCO2017_Val_Eval_Report.json`

Notes:
- The script requires a user-supplied baseline TFLite path (`--baseline-model`).
- `--num-threads` is optional. When omitted, TFLite uses its runtime default.
- mAP is computed from detections with `score >= 0.0` (all model outputs retained by postprocess + max detections cap).
- Noise metrics are reported side-by-side at thresholds `0.05`, `0.1`, and `0.3`.

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
For TFLite I/O contract details, see `docs\android-export-contract.md`.

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

## Generate Android golden sample (detector)

Generate a stable JSON sample for one image:

```powershell
python -m owli_train golden detect --model work\runs\<run_id>\artifacts\model.tflite --image data\coco128\images\train2017\000000000009.jpg --out work\golden\sample.json
```

WSL equivalent:

```bash
python -m owli_train golden detect --model work/runs/<run_id>/artifacts/model.tflite --image data/coco128/images/train2017/000000000009.jpg --out work/golden/sample.json
```

With split-venv delegation:

```powershell
$env:MODELMAKER_PYTHON_EXE=".\.venv-modelmaker-py39\Scripts\python.exe"
python -m owli_train golden detect --model work\runs\<run_id>\artifacts\model.tflite --image data\coco128\images\train2017\000000000009.jpg --out work\golden\sample.json --score-threshold 0.3 --max-results 20
```

Golden JSON contains:
- input preprocessing contract used (letterbox target size, dtype, normalization, pad value)
- top detections (`class_name`, `score`, `bbox`)
- model metadata snapshot (from `*.tflite.meta.json` when present)
- inspect snapshot (operators and I/O tensors)

Android integration should follow the same contract described in `docs/android-contract.md`.

## COCO128 E2E Smoke Script

Run full COCO128 smoke flow (download -> convert -> split -> CSV -> train -> inspect):

```powershell
.\scripts\e2e_coco128_smoke.ps1
```

WSL equivalent:

```bash
bash scripts/e2e_coco128_smoke.sh
```

Script behavior:
- Download URL: `https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip`
- Uses CLI commands:
  - `dataset import yolo`
  - `dataset split`
  - `dataset export modelmaker-csv`
  - `train efficientdet --variant lite2 --max-steps 1`
  - `inspect tflite`

Writes only to:
- `data\` (download + extracted dataset)
- `work\` (generated datasets + runs/artifacts)

Cleanup examples:

```powershell
Remove-Item -Path data\coco128, data\coco128_extract, work\datasets\coco128 -Recurse -Force
Remove-Item -Path work\runs\<run_id> -Recurse -Force
```

```bash
rm -rf data/coco128 data/coco128_extract work/datasets/coco128
rm -rf work/runs/<run_id>
```
