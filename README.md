# Owli Detector Training Tool (Keras + COCO) - Base Project

Local-first Python tooling to:
- validate and normalize COCO object-detection datasets,
- create reproducible train/val/test splits,
- merge multiple COCO sources (GT + pseudo labels) with deterministic mapping,
- materialize merged image trees into a single images root,
- train a KerasCV-based detector,
- evaluate detector runs with COCO mAP reports,
- export to TFLite for Android offline inference,
- (optional later) add segmentation and unified evaluation reports.

## Quickstart (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements\dev.txt
python -m ruff format .
python -m ruff check .
python -m pytest
python -m owli_train --help
```

## Quickstart (WSL2 / Ubuntu)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements/dev.txt
python -m ruff format .
python -m ruff check .
python -m pytest
python -m owli_train --help
```

## CLI examples

```powershell
python -m owli_train dataset validate --coco tests\data\coco_min.json
python -m owli_train dataset normalize --coco tests\data\coco_min.json --out work\normalized\instances.json
python -m owli_train dataset split --coco tests\data\coco_min.json --out-dir work\splits --seed 1337
python -m owli_train dataset merge coco --manifest configs\merge_coco.yaml --out work\datasets\merged\instances.json
python -m owli_train dataset materialize-images --coco work\datasets\merged\instances.json --merge-manifest configs\merge_coco.yaml --out-images-dir work\datasets\merged\images --out-coco work\datasets\merged\instances.materialized.json
```

Training baseline (requires TensorFlow + KerasCV):

```powershell
pip install -r requirements\keras.txt
python -m owli_train train detect --config configs\train_detector.yaml --max-steps 1
```

EfficientDet Model Maker backend:

```powershell
pip install -r requirements\modelmaker.txt
python -m owli_train train efficientdet --config configs\efficientdet_lite2_coco128.yaml --max-steps 1
```

Use a dedicated venv for Model Maker dependencies.
On WSL, use Python 3.9 for the Model Maker venv.
For the WSL smoke script with split envs:
`MODELMAKER_PYTHON_EXE=.venv-modelmaker-py39/bin/python bash scripts/e2e_coco128_smoke.sh`
For COCO val2017 compare bootstrap (dataset + optional baseline model):
`bash scripts/fetch_coco2017_val.sh --coco-root data/coco2017 --with-baseline`

See `docs/runbook.md` for end-to-end dataset operations.
See `docs/wsl-setup.md` for WSL-specific setup and performance notes.
See `docs/BA_v2_Hazard_Labelset.md` for the preferred hazard-centered product ontology.
See `docs/BA_v1_Labelset.md` for the historical verified BA-v1 interim contract.
See `docs/MVP_Training_Plan.md` for the current transition from the historical BA-v1 baseline to the preferred BA-v2 hazard path.

GPU note (RTX-3060 on Windows): TensorFlow GPU is generally smoothest in WSL2.

## WebUI (Phase 2, read-only + safe job launchers)

The first local WebUI lives under `src/owli_train/webui/` and adds a small FastAPI +
Uvicorn control surface over the existing repo state. It still does not replace the CLI,
but it now includes a small whitelist of lightweight dataset-prep jobs with persistent
status and log visibility.

Use the main tooling venv for this UI, not the dedicated Model Maker or teacher venvs.
If you already installed `requirements/dev.txt`, the WebUI dependencies are included.
Otherwise install `requirements/webui.txt`.

WSL:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m uvicorn owli_train.webui.app:app --host 127.0.0.1 --port 8000 --reload
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m uvicorn owli_train.webui.app:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/`.

Current pages:
- `/` dashboard for repo docs, contracts, artifact roots, detected datasets, and detected runs
- `/contracts` for BA-v1 and BA-v2 ontology display
- `/artifacts` for curated dataset/run/config path visibility
- `/jobs` for job list, job detail links, and safe launchers for selected CLI commands

Phase-2 job whitelist:
- `dataset validate`
- `dataset split`
- `dataset merge coco`
- `dataset export modelmaker-csv`
- `dataset materialize-images` via merge manifest only

Deliberately not supported from the WebUI yet:
- training jobs
- teacher pseudo-labeling
- eval / golden chains
- arbitrary shell commands

See `docs/webui.md` for the local start path and current scope.
