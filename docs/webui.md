# WebUI Phase 1

## Purpose
- Add a small local browser entry point over the existing repo and artifact layout.
- Keep the current CLI and filesystem pipeline as the source of truth.
- Prepare a clean seam for later job visibility and job control without building that now.

## Current scope
- FastAPI + Uvicorn app with server-rendered templates
- read-only dashboard for repo docs, label contracts, artifact roots, datasets, and runs
- read-only contracts page for BA-v1 and BA-v2 hazard ontology visibility
- read-only artifacts page for curated dataset/run/config path visibility

## Explicit non-goals in phase 1
- no training start/stop actions
- no dataset mutation
- no annotation UI
- no auth or users
- no database requirement
- no worker or queue system
- no replacement of existing CLI commands

## Repo location
- `src/owli_train/webui/`

## Local start
Use the main tooling venv, not the dedicated Model Maker or teacher venvs.

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

Default port:
- `8000`

Main routes:
- `/`
- `/contracts`
- `/artifacts`

## Data sources used by the UI
- `configs/label_contracts/*.yaml`
- curated docs in `README.md` and `docs/`
- curated artifact roots such as `work/datasets`, `work/runs`, `work/splits`, `work/reports`, `outputs`
- config references under `configs/*.yaml`

Missing local artifact paths are shown as missing instead of raising UI errors.
