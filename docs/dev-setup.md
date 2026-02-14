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

GPU note: TensorFlow GPU is generally easiest in WSL2 (Ubuntu).
