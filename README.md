# Owli Detector Training Tool (Keras + COCO) - Base Project

Local-first Python tooling to:
- validate and normalize COCO object-detection datasets,
- create reproducible train/val/test splits,
- train a KerasCV-based detector,
- evaluate detector runs with COCO mAP reports,
- export to TFLite for Android (next milestone),
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

## CLI examples

```powershell
python -m owli_train dataset validate --coco tests\data\coco_min.json
python -m owli_train dataset normalize --coco tests\data\coco_min.json --out work\normalized\instances.json
python -m owli_train dataset split --coco tests\data\coco_min.json --out-dir work\splits --seed 1337
```

Training baseline (requires TensorFlow + KerasCV):

```powershell
pip install -r requirements\keras.txt
python -m owli_train train detect --config configs\train_detector.yaml --max-steps 1
```

See `docs/runbook.md` for end-to-end dataset operations.

GPU note (RTX-3060 on Windows): TensorFlow GPU is generally smoothest in WSL2.
