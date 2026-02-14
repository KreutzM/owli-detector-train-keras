# Owli Detector Training Tool (Keras + COCO) — Base Project

Local-first Python tooling to:
- validate / normalize COCO object-detection datasets,
- create reproducible train/val/test splits,
- (next milestones) train a Keras-based detector and export to TFLite for Android,
- (optional later) add a segmentation model and a unified evaluation report.

Quickstart (Windows / PowerShell):

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

GPU note (RTX-3060 on Windows): TensorFlow GPU is generally smoothest in WSL2.
