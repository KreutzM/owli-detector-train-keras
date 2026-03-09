# WebUI Phase 4

## Purpose
- Add a small local browser entry point over the existing repo and artifact layout.
- Keep the current CLI and filesystem pipeline as the source of truth.
- Add a first practical control seam for a few small whitelisted CLI jobs without building a full queue or worker platform.
- Add useful dataset, run, eval, and golden detail pages so the UI works as a diagnosis surface, not only as a launcher list.
- Add a first small bridge from the WebUI into local FiftyOne for visual dataset inspection.

## Current scope
- FastAPI + Uvicorn app with server-rendered templates
- read-only dashboard for repo docs, label contracts, artifact roots, datasets, and runs
- read-only contracts page for BA-v1 and BA-v2 hazard ontology visibility
- read-only artifacts page for curated dataset/run/config path visibility
- dataset detail pages with:
  - COCO summary counts
  - class distribution
  - split counts
  - QC summary if `qc_report.json` exists
  - related config references where the dataset path is referenced
  - a local `Open in FiftyOne` action when the dataset has a usable COCO file and `<dataset>/images`
- run detail pages with:
  - artifact and report listings
  - config snapshot visibility
  - links into eval and golden detail pages
- eval detail pages with:
  - global metrics
  - summary counts
  - per-class metrics when present in JSON reports
  - nearby eval-report references in the same run
  - a local FiftyOne action when the eval JSON contains repo-local `coco_path` and `images_dir`
- golden detail pages with:
  - image/model summary metadata
  - contract and inspect-TFLite fields
  - detection tables
- jobs page with:
  - recent job list
  - job detail pages
  - persisted job status and logs
  - small launch forms for selected dataset-prep commands

## Supported job types in phase 4
- `dataset validate`
- `dataset split`
- `dataset merge coco`
- `dataset export modelmaker-csv`
- `dataset materialize-images` with manifest-backed source resolution only

## FiftyOne scope in phase 4
- supported now:
  - dataset detail -> open supported COCO dataset in local FiftyOne
  - eval detail -> open the dataset referenced by eval JSON when `coco_path` and `images_dir` resolve inside the repo
  - one small local launch bridge managed by the WebUI process
- intentionally not supported yet:
  - importing predictions or eval overlays into FiftyOne
  - filter/state sync between WebUI and FiftyOne
  - embedded FiftyOne UI inside the WebUI
  - multi-user or durable session management

## Explicit non-goals in phase 4
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

Optional FiftyOne install in the same venv:

WSL:

```bash
source .venv/bin/activate
pip install -r requirements/fiftyone.txt
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements\fiftyone.txt
```

Default port:
- `8000`

Main routes:
- `/`
- `/contracts`
- `/artifacts`
- `/datasets/view?path=...`
- `/runs/view?path=...`
- `/evals/view?path=...`
- `/goldens/view?path=...`
- `/fiftyone/open?source=...&path=...`
- `/jobs`
- `/jobs/{job_id}`

## Data sources used by the UI
- `configs/label_contracts/*.yaml`
- curated docs in `README.md` and `docs/`
- curated artifact roots such as `work/datasets`, `work/runs`, `work/splits`, `work/reports`, `outputs`
- config references under `configs/*.yaml`
- persisted job files and logs under `work/webui/jobs/`
- eval and golden JSON artifacts under `work/runs/*/reports/` when present
- QC reports such as `work/datasets/*/qc_report.json` when present

## FiftyOne runtime model
- FiftyOne is optional and is only touched when the launch route is used
- the WebUI starts a small local Python subprocess for the selected dataset
- the subprocess imports a COCO dataset into FiftyOne and starts the local FiftyOne app
- the current implementation prefers explicit repo-local paths over inference magic
- unsupported or incomplete datasets are explained in the detail page instead of failing the whole UI
- current dataset requirements:
  - dataset detail path must expose a COCO file plus `<dataset>/images`
  - eval detail path must expose repo-local `coco_path` and `images_dir` in JSON

## Job runtime model
- jobs are launched only from a small internal whitelist
- execution uses Python subprocesses, not shell strings
- each job stores status, timestamps, exit code, parameters, and expected artifact paths
- logs are written to `work/webui/jobs/logs/`
- job records persist across app restarts via file-backed JSON metadata

Missing local artifact paths are shown as missing instead of raising UI errors.
