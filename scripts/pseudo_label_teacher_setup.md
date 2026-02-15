# Teacher Pseudo-Label Setup (TF Hub / SavedModel)

## 1) Create dedicated teacher venv

PowerShell:

```powershell
python -m venv .venv-teacher
.\.venv-teacher\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements\teacher.txt
$env:TEACHER_PYTHON_EXE=".\.venv-teacher\Scripts\python.exe"
```

WSL:

```bash
python3 -m venv .venv-teacher
source .venv-teacher/bin/activate
python -m pip install --upgrade pip
pip install -r requirements/teacher.txt
export TEACHER_PYTHON_EXE=.venv-teacher/bin/python
```

## 2) GPU check (teacher env)

```bash
$TEACHER_PYTHON_EXE -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 3) Generate pseudo labels (COCO-80)

```bash
python -m owli_train dataset pseudo-label coco \
  --images-dir data/ba/images \
  --out work/pseudo/ba_pseudo_coco80.json \
  --batch-size 16 \
  --score-threshold 0.6 \
  --max-detections-per-image 50
```

Outputs:
- `work/pseudo/ba_pseudo_coco80.json`
- `work/pseudo/ba_pseudo_coco80.report.json`
