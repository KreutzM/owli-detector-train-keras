# AGENTS.md - Guidance for OpenAI Codex CLI (gpt-5.3-codex)

## Goal
Prepare COCO-format datasets, train Keras-based vision models (detector now, optional segmenter later), evaluate, and export TFLite artifacts for Android offline inference.

## Non-negotiables
- Tests must be offline (no network), lightweight, and GPU-free.
- Do not commit datasets/checkpoints/large artifacts.
- PowerShell-first docs/scripts on Windows; provide WSL/bash equivalents where relevant.
- For every non-trivial task, create or update `docs/reviews/Codex-Task-Report_last.md` from `docs/review-templates/Codex-Task-Report.md` and keep the exact template section structure.
- After meaningful changes run:
  - python -m ruff format .
  - python -m ruff check .
  - python -m pytest

## Workflow
1) Read README.md + relevant docs/.
2) Plan briefly (3-7 bullets), then implement in small steps.
3) Add/adjust tests for behavior changes.
4) Update docs if user-facing behavior changes.
5) Commit small, thematic commits: feat/fix/docs/test/refactor/chore.
