# Codex Memory (Session Context for WSL2 Restart)

Last updated: 2026-02-14
Current branch at handoff: `main`
Git working tree at handoff: clean

## Session Summary

This session covered milestone-style work across dataset tooling, training, evaluation, export, Android compatibility, EfficientDet Model Maker integration, and WSL2 compatibility hardening.

Main outcomes now present in `main`:
- COCO dataset CLI foundation (`validate`, `normalize`, `split`) and tests.
- KerasCV detector training pipeline (`train detect`) with smoke flags.
- Evaluation pipeline (`eval detect`) with report artifacts.
- TFLite export/bench/inspect commands.
- Android compatibility checks (`builtins` vs `Select TF Ops`).
- EfficientDet-Lite Model Maker backend (`train efficientdet`) plus dataset adapters.
- COCO128 E2E PowerShell script and repo hygiene rules.
- WSL2 compatibility additions (LF policy, WSL setup docs, WSL E2E shell script).

## Important Merge/Commit Context

Recent commits merged into `main`:
- `6d3bea4` docs: add WSL setup and update runbook/readme
- `5b35a6d` chore: align modelmaker requirements for WSL setup
- `91ddc23` feat: add WSL coco128 e2e smoke script
- `81f08a2` chore: add .gitattributes for cross-platform eol
- `aaa3223` merge: `feat/m8-e2e-smoke-coco128`

Earlier milestone commits on `main` include:
- EfficientDet Model Maker backend + YOLO->COCO and COCO->ModelMaker CSV adapters
- COCO128 smoke pipeline + docs/tests
- TFLite export/bench/inspect and Android compatibility path

## WSL2-Specific Decisions

Implemented:
- `.gitattributes` with LF-by-default:
  - `* text=auto eol=lf`
  - `*.sh text eol=lf`
  - `*.toml text eol=lf`
  - `*.md text eol=lf`
  - `*.ps1 text eol=crlf`
- New WSL-native smoke entry point:
  - `scripts/e2e_coco128_smoke.sh`
  - Runs: download -> import yolo -> split -> export modelmaker-csv -> train efficientdet (`--max-steps 1`) -> inspect tflite
  - Writes only under `data/` and `work/`
- New doc:
  - `docs/wsl-setup.md`
- Updated docs:
  - `README.md`
  - `docs/runbook.md`
  - `docs/dev-setup.md`

## Model Maker Dependency Strategy

Key decision:
- Keep Model Maker dependencies separate from KerasCV detector dependencies.

Current `requirements/modelmaker.txt`:
- Standalone base runtime deps (`typer`, `rich`, `PyYAML`, `Pillow`, `tqdm`)
- Pinned compatibility stack:
  - `numpy>=1.22,<1.24`
  - `tensorflow==2.13.1`
  - `tflite-model-maker==0.4.3`

Reason:
- Avoid repeated conflicts between modern `tensorflow/keras-cv` stack and legacy Model Maker constraints.

Operational recommendation:
- Use separate venvs:
  - `.venv` for regular dev + KerasCV detector
  - `.venv-modelmaker` for `train efficientdet` path

## Known Runtime Notes

- EfficientDet Model Maker path required `TF_USE_LEGACY_KERAS=1` for successful smoke run in this environment.
- This is set in both E2E scripts:
  - `scripts/e2e_coco128_smoke.ps1`
  - `scripts/e2e_coco128_smoke.sh`
- COCO128 archive currently used:
  - `https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip`
- Some extracted label files can reference missing images; scripts defensively remove dangling labels before import.

## Successful Smoke Evidence in Session

PowerShell E2E smoke completed successfully with:
- run dir: `work/runs/20260214-175613`
- artifact: `work/runs/20260214-175613/artifacts/model.tflite`
- inspect output included:
  - `builtin_ops_only: true`

Note:
- `work/` artifacts are local and not part of git history.

## CLI Surface Available

Primary commands present:
- `python -m owli_train dataset validate`
- `python -m owli_train dataset normalize`
- `python -m owli_train dataset split`
- `python -m owli_train dataset import yolo`
- `python -m owli_train dataset export modelmaker-csv`
- `python -m owli_train train detect`
- `python -m owli_train train efficientdet`
- `python -m owli_train eval detect`
- `python -m owli_train export tflite`
- `python -m owli_train bench tflite`
- `python -m owli_train inspect tflite`

## Repository Hygiene State

- `data/`, `work/`, `outputs/` are ignored in `.gitignore`.
- Temp patterns are ignored (`tmp/`, `TMP/`, `.tmp/`, recursive variants).
- Ruff/Pytest exclude temp/artifact dirs via `pyproject.toml`.

## WSL2 Fresh-Start Checklist

1. Clone into Linux filesystem, not `/mnt/c`:
   - Example: `~/src/owli-detector-train-keras`
2. Create regular dev env:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements/dev.txt`
   - `pip install -r requirements/keras.txt` (if `train detect`)
3. Optional Model Maker env:
   - `python3 -m venv .venv-modelmaker`
   - `source .venv-modelmaker/bin/activate`
   - `pip install -r requirements/modelmaker.txt`
4. Sanity checks:
   - `python -m ruff check .`
   - `python -m pytest`
   - `python -m owli_train --help`
5. WSL E2E smoke:
   - `bash scripts/e2e_coco128_smoke.sh`

## What Was Intentionally Not Included in the WSL Compatibility Commits

Two local edits from debugging were explicitly kept out of the WSL compatibility merge and restored before merge:
- `scripts/e2e_coco128_smoke.ps1`
- `src/owli_train/training/modelmaker_efficientdet.py`

Only the WSL-focused and requirements/docs changes above were merged into `main`.
