# Codex Start (PowerShell)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements\dev.txt
```

## Sanity checks

```powershell
python -m ruff format .
python -m ruff check .
python -m pytest
python -m owli_train --help
```

## Milestone 1 smoke commands

```powershell
python -m owli_train dataset validate --coco tests\data\coco_min.json
python -m owli_train dataset normalize --coco tests\data\coco_min.json --out work\normalized\instances.json
python -m owli_train dataset split --coco tests\data\coco_min.json --out-dir work\splits --seed 1337
```

## Optional with image checks

```powershell
python -m owli_train dataset validate --coco data\instances.json --images-dir data\images
```
