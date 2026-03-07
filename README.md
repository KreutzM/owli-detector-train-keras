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
See `docs/BA_v1_Labelset.md` for the current BA-v1 product label contract and dataset priorities.

GPU note (RTX-3060 on Windows): TensorFlow GPU is generally smoothest in WSL2.
