# Codex Task Report

## Ziel
- Die bestehende WebUI um echte Analyse- und Detailansichten fuer Datasets, Runs, Eval-Ergebnisse und Golden-Artefakte erweitern.
- Die Diagnose- und Vergleichbarkeit der vorhandenen Pipeline-Artefakte verbessern, ohne die CLI zu ersetzen oder eine grosse Plattform zu bauen.
- Die bestehende kleine Job-Steuerung beibehalten und die WebUI sinnvoll zwischen Analyse und sicherer Steuerung ausbalancieren.

## Was wurde geaendert?
- WebUI-App um vier neue Detailrouten erweitert:
  - `src/owli_train/webui/app.py`
- Reader-/Service-Layer um kleine Dateisystem-Reader fuer Dataset-, Run-, Eval- und Golden-Details erweitert:
  - `src/owli_train/webui/readers.py`
- Neue kleine View-Modelle fuer Detailseiten, Metriken und Tabellen ergaenzt:
  - `src/owli_train/webui/models.py`
- Bestehende Templates fuer Navigation und Links erweitert:
  - `src/owli_train/webui/templates/base.html`
  - `src/owli_train/webui/templates/dashboard.html`
  - `src/owli_train/webui/templates/artifacts.html`
  - `src/owli_train/webui/templates/job_detail.html`
- Neue server-rendered Detail-Templates angelegt:
  - `src/owli_train/webui/templates/dataset_detail.html`
  - `src/owli_train/webui/templates/run_detail.html`
  - `src/owli_train/webui/templates/eval_detail.html`
  - `src/owli_train/webui/templates/golden_detail.html`
- Test-Fixture um kleine QC-, Eval- und Golden-Beispielartefakte erweitert:
  - `tests/webui_test_utils.py`
- Kleine Reader- und Route-Tests fuer die neuen Detailansichten ergaenzt:
  - `tests/test_webui_reader.py`
  - `tests/test_webui_app.py`
- Doku auf Phase 3 aktualisiert:
  - `README.md`
  - `docs/webui.md`
  - `docs/runbook.md`

## Was wurde wirklich verifiziert?
- Statisch geprueft:
  - `README.md`
  - `docs/webui.md`
  - `docs/runbook.md`
  - `src/owli_train/webui/*`
  - relevante Report-/Artefaktquellen:
    - `src/owli_train/data/balance_coco.py`
    - `src/owli_train/data/merge_coco.py`
    - `src/owli_train/data/obstacle_dataset.py`
    - `src/owli_train/eval/detect.py`
    - `src/owli_train/eval/efficientdet_tflite.py`
    - `src/owli_train/golden/detect.py`
  - `docs/reviews/Codex-Task-Report_last.md` (vorheriger Stand)
- Inhaltlich verifiziert:
  - Dataset-Details lesen COCO-Zaehlungen, Klassenverteilung, Split-Dateien und QC-Reports robust read-only aus.
  - Run-Details lesen bekannte Artefakt-/Report-/Snapshot-Dateien aus `work/runs/*`.
  - Eval-Details lesen globale JSON-Metriken, Summary-Counts und per-class Kennzahlen, falls vorhanden.
  - Golden-Details lesen Metadaten, Contract-Felder und Detections aus Golden-JSON-Dateien.
  - Fehlende oder nicht vorhandene Artefakte fuehren zu leeren Zustaenden oder `404`, nicht zu UI-Crashes.
- Real ausgefuehrt:
  - neue WebUI-Reader- und Route-Tests
  - Ruff Format / Ruff Check / Pytest
  - lokaler Uvicorn-Start gegen das echte Repo
  - echte HTTP-GETs auf Dashboard/Artifacts und die neuen Detailrouten gegen das echte Repo
  - lokaler Uvicorn-Start gegen ein temporäres Sample-Repo mit Mini-Artefakten
  - echte HTTP-GETs auf Dataset-/Run-/Eval-/Golden-Detailseiten mit `200` gegen das Sample-Repo

## Tests
- `python -m pytest tests/test_webui_reader.py tests/test_webui_app.py tests/test_webui_jobs.py`
  - Exit-Code: `0`
  - Ergebnis: `12 passed in 2.09s`
- `python -m ruff check src/owli_train/webui tests/test_webui_app.py tests/test_webui_reader.py tests/webui_test_utils.py`
  - Exit-Code: `1`
  - Ergebnis: Import-Sortierung in `src/owli_train/webui/readers.py` beanstandet
- `python -m ruff format .`
  - Exit-Code: `0`
  - Ergebnis: `2 files reformatted, 76 files left unchanged`
- `python -m ruff check .`
  - Exit-Code: `1`
  - Ergebnis: derselbe Import-Block in `src/owli_train/webui/readers.py` blieb offen
- `python -m pytest`
  - Exit-Code: `0`
  - Ergebnis: `177 passed, 5 skipped in 6.16s`
- `python -m ruff check src/owli_train/webui/readers.py --fix`
  - Exit-Code: `0`
  - Ergebnis: `1` Ruff-Fix automatisch angewendet
- `python -m ruff check .`
  - Exit-Code: `0`
  - Ergebnis: `All checks passed!`
- `python -m pytest tests/test_webui_reader.py tests/test_webui_app.py`
  - Exit-Code: `0`
  - Ergebnis: `6 passed in 0.65s`
- `PYTHONPATH=src python -m uvicorn owli_train.webui.app:app --host 127.0.0.1 --port 8000`
  - Exit-Code: `0` nach sauberem `CTRL+C`
  - Ergebnis: App startete lokal gegen das echte Repo
- 
  ```bash
  python - <<'PY'
  import httpx

  base = 'http://127.0.0.1:8000'
  paths = [
      '/',
      '/artifacts',
      '/datasets/view?path=work/datasets/demo-dataset',
      '/runs/view?path=work/runs/20260309-123000-demo',
      '/evals/view?path=work/runs/20260309-123000-demo/reports/eval_demo.json',
      '/goldens/view?path=work/runs/20260309-123000-demo/reports/golden_obstacle4.json',
  ]
  with httpx.Client(base_url=base, timeout=10.0) as client:
      for path in paths:
          response = client.get(path)
          print(path, response.status_code, len(response.text))
  PY
  ```
  - Exit-Code: `0`
  - Ergebnis:
    - `/` -> `200`
    - `/artifacts` -> `200`
    - neue Detailrouten im echten Repo -> `404`, weil die konkret angefragten Sample-Artefakte dort aktuell nicht vorhanden sind
- 
  ```bash
  PYTHONPATH=src:. python - <<'PY'
  import tempfile
  from pathlib import Path

  import uvicorn

  from owli_train.webui.app import create_app
  from tests.webui_test_utils import build_sample_repo

  repo_root = build_sample_repo(Path(tempfile.mkdtemp(prefix='owli-webui-sample-')))
  uvicorn.run(create_app(repo_root=repo_root), host='127.0.0.1', port=8001)
  PY
  ```
  - Exit-Code: `0` nach sauberem `CTRL+C`
  - Ergebnis: App startete lokal gegen ein temporaeres Sample-Repo mit Mini-Artefakten
- 
  ```bash
  python - <<'PY'
  import httpx

  base = 'http://127.0.0.1:8001'
  paths = [
      '/',
      '/artifacts',
      '/datasets/view?path=work/datasets/demo-dataset',
      '/runs/view?path=work/runs/20260309-123000-demo',
      '/evals/view?path=work/runs/20260309-123000-demo/reports/eval_demo.json',
      '/goldens/view?path=work/runs/20260309-123000-demo/reports/golden_obstacle4.json',
  ]
  with httpx.Client(base_url=base, timeout=10.0) as client:
      for path in paths:
          response = client.get(path)
          print(path, response.status_code, len(response.text))
  PY
  ```
  - Exit-Code: `0`
  - Ergebnis:
    - `/` -> `200`
    - `/artifacts` -> `200`
    - `/datasets/view?...` -> `200`
    - `/runs/view?...` -> `200`
    - `/evals/view?...` -> `200`
    - `/goldens/view?...` -> `200`
- `python -m pytest`
  - Exit-Code: `0`
  - Ergebnis: `177 passed, 5 skipped in 11.75s`

## Relevante Run-Kommandos
- WSL2-Start der WebUI:
```bash
source .venv/bin/activate
PYTHONPATH=src python -m uvicorn owli_train.webui.app:app --host 127.0.0.1 --port 8000 --reload
```
- Technischer Sample-Repo-Start fuer lokale Renderpruefung der Detailseiten:
```bash
PYTHONPATH=src:. python - <<'PY'
import tempfile
from pathlib import Path

import uvicorn

from owli_train.webui.app import create_app
from tests.webui_test_utils import build_sample_repo

repo_root = build_sample_repo(Path(tempfile.mkdtemp(prefix='owli-webui-sample-')))
uvicorn.run(create_app(repo_root=repo_root), host='127.0.0.1', port=8001)
PY
```

## Offene Risiken
- Die Detailseiten basieren bewusst auf bekannten Dateiformaten; stark abweichende oder spaeter geaenderte Report-Schemata werden aktuell nur begrenzt interpretiert.
- Im echten Repo koennen Detailrouten nur dann Inhalte rendern, wenn passende Artefakte unter den bekannten `work/`-Pfade auch wirklich vorhanden sind.
- Die Eval-Vergleichbarkeit bleibt absichtlich klein; es gibt noch keine dedizierte mehrseitige Vergleichsansicht ueber mehrere Runs hinweg.

## Naechster sinnvoller Schritt
- Ergaenze als naechsten kleinen Schritt eine kuratierte Multi-Run-Vergleichsansicht fuer Eval-Reports, damit wichtige Stage-Varianten direkt in der WebUI nebeneinander lesbar werden.
