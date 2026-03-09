# WebUI Phase 2

## Purpose
- Add a small local browser entry point over the existing repo and artifact layout.
- Keep the current CLI and filesystem pipeline as the source of truth.
- Add a first practical control seam for a few small whitelisted CLI jobs without building a full queue or worker platform.

## Current scope
- FastAPI + Uvicorn app with server-rendered templates
- read-only dashboard for repo docs, label contracts, artifact roots, datasets, and runs
- read-only contracts page for BA-v1 and BA-v2 hazard ontology visibility
- read-only artifacts page for curated dataset/run/config path visibility
- jobs page with:
  - recent job list
  - job detail pages
  - persisted job status and logs
  - small launch forms for selected dataset-prep commands

## Supported job types in phase 2
- `dataset validate`
- `dataset split`
- `dataset merge coco`
- `dataset export modelmaker-csv`
- `dataset materialize-images` with manifest-backed source resolution only

## Explicit non-goals in phase 2
- no training start/stop actions
- no heavy GPU jobs
- no teacher pseudo-labeling
- no annotation UI
- no auth or users
- no database requirement
- no Redis / Celery / RQ stack
- no arbitrary shell command input
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
- `/jobs`
- `/jobs/{job_id}`

## Data sources used by the UI
- `configs/label_contracts/*.yaml`
- curated docs in `README.md` and `docs/`
- curated artifact roots such as `work/datasets`, `work/runs`, `work/splits`, `work/reports`, `outputs`
- config references under `configs/*.yaml`
- persisted job files and logs under `work/webui/jobs/`

## Job runtime model
- jobs are launched only from a small internal whitelist
- execution uses Python subprocesses, not shell strings
- each job stores status, timestamps, exit code, parameters, and expected artifact paths
- logs are written to `work/webui/jobs/logs/`
- job records persist across app restarts via file-backed JSON metadata

Missing local artifact paths are shown as missing instead of raising UI errors.
