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

## Merge COCO sources (GT + pseudo labels)

Merge multiple COCO files via a manifest. This is useful to combine:
- hand-labeled GT datasets
- COCO80 pseudo labels from `dataset pseudo-label coco`
- per-source label mappings into one canonical taxonomy

Example manifest (`configs\merge_coco.yaml`):

```yaml
sources:
  - name: custom_gt
    coco: work/datasets/custom/instances.json
    images_dir: data/custom/images
    label_map: configs/custom_to_canonical.yaml
  - name: custom_pseudo_coco80
    coco: work/pseudo/custom_pseudo_coco80.json
    images_dir: data/custom/images
    pseudo: true
    score_threshold: 0.6
settings:
  same_class_iou: 0.75
  pseudo_block_iou: 0.6
  allow_duplicate_file_names: false
```

Run merge:

```powershell
python -m owli_train dataset merge coco --manifest configs\merge_coco.yaml --out work\datasets\merged\instances.json
```

Outputs:
- merged COCO: `--out`
- merge report JSON: `--report-out` or default `<out>.report.json`

Notes:
- GT has priority over pseudo labels in overlapping regions.
- Pseudo labels below `score_threshold` are dropped per pseudo source.
- Duplicate `file_name` across different source namespaces is rejected by default to avoid accidental collisions. Use per-source `file_name_prefix` in manifest when needed.

## Materialize merged images into one root

After merge, materialize all referenced images into a single directory so downstream
commands can use one `--images-dir`.

Using the same merge manifest (recommended):

```powershell
python -m owli_train dataset materialize-images --coco work\datasets\merged\instances.json --merge-manifest configs\merge_coco.yaml --out-images-dir work\datasets\merged\images --out-coco work\datasets\merged\instances.materialized.json --mode auto
```

Alternative without manifest (explicit source roots):

```powershell
python -m owli_train dataset materialize-images --coco work\datasets\merged\instances.json --source-images-dir data\setA\images --source-images-dir data\setB\images --out-images-dir work\datasets\merged\images --out-coco work\datasets\merged\instances.materialized.json --mode copy
```

Modes:
- `auto`: try symlink first, fallback to copy.
- `symlink`: symlink only (fail on symlink errors).
- `copy`: copy files.

Outputs:
- materialized image tree: `--out-images-dir`
- COCO with normalized relative file names: `--out-coco`

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

Docker GPU fallback for legacy Model Maker stack (recommended when local Model Maker venv cannot see GPU):

```powershell
.\scripts\modelmaker_gpu_docker.ps1 build
.\scripts\modelmaker_gpu_docker.ps1 gpu-check
.\scripts\modelmaker_gpu_docker.ps1 run train efficientdet configs\efficientdet_lite2_coco2017.yaml --max-steps 500 --subset-seed 1337 --require-gpu
```

WSL/bash equivalent:

```bash
bash scripts/modelmaker_gpu_docker.sh build
bash scripts/modelmaker_gpu_docker.sh gpu-check
bash scripts/modelmaker_gpu_docker.sh run -- train efficientdet configs/efficientdet_lite2_coco2017.yaml --max-steps 500 --subset-seed 1337 --require-gpu
```

The bash runner auto-detects `.venv-modelmaker-py39/bin/python` when present.

If you want to run with a mounted external Model Maker venv Python inside the container:

```bash
export MODELMAKER_DOCKER_PYTHON_EXE=/workspace/.venv-modelmaker-py39/bin/python
export MODELMAKER_DOCKER_EXTRA_MOUNTS="$HOME/.local/share/uv:$HOME/.local/share/uv:ro"
bash scripts/modelmaker_gpu_docker.sh gpu-check
bash scripts/modelmaker_gpu_docker.sh run -- train efficientdet configs/efficientdet_lite2_coco2017.yaml --max-steps 1 --subset-seed 1337 --require-gpu
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
- The training pipeline canonicalizes class first-occurrence order from `label_map_json` before Model Maker load (without globally grouping rows by class).
- Training now fails fast if a class from `data.label_map_json` is missing from `TRAIN` rows in `data.csv`, because Model Maker would otherwise drop that class from the exported Lite label set.
- For intentional non-product runs with a reduced TRAIN class set, set `train.allow_missing_train_classes: true` in the EfficientDet config.
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

Batch-size note:
- If the teacher signature exposes a fixed batch dimension (for example `1`), the CLI now auto-adjusts to that batch size and records the effective batch size in the report.
- If a multi-image batch still fails at inference time, rerun with `--batch-size 1`.

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
python -m owli_train golden detect --model work\runs\<run_id>\artifacts\model.tflite --image data\coco128\images\train2017\000000000009.jpg --out work\golden\sample.json --score-threshold 0.3 --max-results 20 --num-threads 8
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

## Obstacle4 (DS1) quick recipe

Obstacle4 is the current verified baseline, not the whole forward-looking MVP path.
For the planned multi-source MVP assembly path, see [MVP_Training_Plan.md](./MVP_Training_Plan.md).

Dataset:
- Mendeley DOI: `10.17632/xwhnp82rhk.1`
- Expected layout after extraction: `train/images`, `train/labels`, `valid/images`, `valid/labels`, `data.yaml`

Core files used by the DS1 pipeline:
- Label map: `configs/label_maps/obstacle4_to_ba.yaml`
- Merge manifest: `configs/merge_obstacle4_gt_pseudo.yaml`
- Training config: `configs/efficientdet_lite2_obstacle4.yaml`
- BA-v1 contract: `configs/label_contracts/ba_v1.yaml`
- Result summary: `docs/Obstacle4_E2E_Results.md`
- Product labelset summary: `docs/BA_v1_Labelset.md`

WSL example flow:

```bash
python -m owli_train dataset import yolo --yolo-dir data/raw/obstacle4/extracted --out work/datasets/obstacle4/instances_raw.json
python -m owli_train dataset normalize --coco work/datasets/obstacle4/instances_raw.json --images-dir data/raw/obstacle4/extracted --label-map configs/label_maps/obstacle4_to_ba.yaml --out work/datasets/obstacle4/instances_gt.json

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train dataset pseudo-label coco \
  --images-dir data/raw/obstacle4/extracted \
  --out work/datasets/obstacle4/pseudo_coco_critical.json \
  --classes person,bicycle,motorcycle,car,bus,truck --score-threshold 0.45 --batch-size 1

python -m owli_train dataset merge coco --manifest configs/merge_obstacle4_gt_pseudo.yaml --out work/datasets/obstacle4/instances_combined.json
python -m owli_train dataset split --coco work/datasets/obstacle4/instances_combined.json --out-dir work/splits/obstacle4 --seed 1337 --ensure-train-class-coverage
python -m owli_train dataset export modelmaker-csv --coco work/datasets/obstacle4/instances_combined.json --images-dir data/raw/obstacle4/extracted --splits-json work/splits/obstacle4/splits.json --out work/datasets/obstacle4/modelmaker.csv

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet --config configs/efficientdet_lite2_obstacle4.yaml --require-gpu

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite --model work/runs/<run_id>/artifacts/model.tflite

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/datasets/obstacle4/instances_combined.json \
  --images-dir data/raw/obstacle4/extracted \
  --model work/runs/<run_id>/artifacts/model.tflite \
  --score-threshold 0.1 --noise-thresholds 0.05,0.1,0.3 --num-threads 8 \
  --out work/runs/<run_id>/reports/eval_efficientdet_tflite.json

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect \
  --model work/runs/<run_id>/artifacts/model.tflite \
  --image data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg \
  --out work/runs/<run_id>/reports/golden_obstacle4.json \
  --score-threshold 0.1 --max-results 20 --num-threads 8
```

For Obstacle4, build `splits.json` from `instances_combined.json`, not GT-only COCO.
The pseudo-label classes (`person`, `bicycle`, `motorcycle`, `car`, `bus`, `truck`) are part of the
product label contract, so `--ensure-train-class-coverage` is the safest way to keep rare merged
classes from falling out of `TRAIN` when they are present in the data.

## Obstacle-Dataset (DS2) prep

- Current status: prepared in-repo, not yet integrated with real local raw data.
- Checked-in mapping stub: `configs/label_maps/obstacle_dataset_to_ba.yaml`
- Integration notes: `docs/Obstacle_Dataset_Integration.md`
- BA-v1 rule for DS2: keep this dataset constrained to the four BA core classes until the local
  source taxonomy is reviewed and explicitly mapped.

## BA MVP multi-source prep

- Primary plan: [MVP_Training_Plan.md](./MVP_Training_Plan.md)
- Fixed class contract: `configs/label_contracts/ba_v1.yaml`
- Verified baseline anchor: `Obstacle4`
- Next BA supplemental sources to review locally after downloads complete:
  - `TACO`
  - `Obstacle-Dataset / OD`
- Mapillary status:
  - local source reviewed
  - BA-filtered converter documented in [Mapillary_Vistas_Integration.md](./Mapillary_Vistas_Integration.md)
- Narrow rehearsal-only source:
  - `COCO replay` for `person`, `bicycle`, `motorcycle`, `car`, `bus`, `truck`

Checked-in prep files:
- `configs/label_maps/mapillary_vistas_to_ba.yaml`
- `configs/label_maps/taco_to_ba.yaml`
- `configs/label_maps/obstacle_dataset_to_ba.yaml`
- `configs/label_maps/coco_replay_to_ba.yaml`

Working rule:
- Keep `Mapillary Vistas`, `TACO`, and `OD` conservative and BA-core-only until the local source
  taxonomies are reviewed from real files.
- Use COCO only as a small replay source for the six BA-v1 rehearsal classes, not as a return to
  broad COCO-80 training.

## Mapillary Vistas -> BA COCO Detection

- Local source root: `data/DataSets/Map`
- Source representation used by the converter:
  - `training/panoptic/panoptic_2018.json`
  - `validation/panoptic/panoptic_2018.json`
  - matching RGB images from `training/images` and `validation/images`
- Current mapping file: `configs/label_maps/mapillary_vistas_to_ba.yaml`
- Integration notes: [Mapillary_Vistas_Integration.md](./Mapillary_Vistas_Integration.md)

Current behavior:
- uses only `training` and `validation`
- ignores `testing`
- keeps only the current narrow BA-relevant class whitelist
- exports resized images with long side capped at `1600`
- writes scaled COCO detection boxes plus `splits.json` and `qc_report.json`
- supports both classic `v1.2` layout and `Map2` `v2.0` panoptic layout
- default stays conservative and prefers `v1.2` when both annotation trees exist

WSL example:
```bash
python -m owli_train dataset import mapillary-vistas \
  --mapillary-dir data/DataSets/Map \
  --out-dir data/processed/mapillary_ba_v1 \
  --label-map configs/label_maps/mapillary_vistas_to_ba.yaml \
  --max-long-side 1600
```

Bounded verification example:
```bash
python -m owli_train dataset import mapillary-vistas \
  --mapillary-dir data/DataSets/Map \
  --out-dir data/processed/mapillary_ba_v1_sample \
  --label-map configs/label_maps/mapillary_vistas_to_ba.yaml \
  --max-long-side 1600 \
  --limit-images-per-split 100
```

Explicit `Map2/v2.0` example:
```bash
python -m owli_train dataset import mapillary-vistas \
  --mapillary-dir data/DataSets/Map2 \
  --out-dir data/processed/mapillary_ba_v2_0 \
  --label-map configs/label_maps/mapillary_vistas_to_ba.yaml \
  --annotation-version v2.0 \
  --max-long-side 1600
```
