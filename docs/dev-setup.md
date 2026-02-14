# Developer setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements\dev.txt
python -m ruff format .
python -m ruff check .
python -m pytest
```

Optional training deps:
```powershell
pip install -r requirements\keras.txt
```

GPU note:
- TensorFlow GPU on Windows is supported via WSL2 (Ubuntu) and is the recommended setup.
- Native Windows TensorFlow for TF>=2.11 is typically CPU-only.
