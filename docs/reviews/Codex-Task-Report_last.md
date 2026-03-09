# Codex Task Report

## Ziel
- Die bestehende WebUI um einen ersten kleinen, lokalen FiftyOne-Hook erweitern.
- Geeignete COCO-Datasets aus der Dataset-Detailseite direkt in FiftyOne oeffnen koennen.
- Optional einen zweiten kleinen Oeffnungspfad ueber eval-verknuepfte Dataset-Referenzen anbieten.
- Die bestehende WebUI-Struktur beibehalten, ohne FiftyOne voll einzubetten oder neue Job-/Session-Plattformen zu bauen.

## Was wurde geändert?
- Kleinen optionalen FiftyOne-Service fuer die WebUI eingefuehrt:
  - `src/owli_train/webui/fiftyone.py`
  - `src/owli_train/webui/fiftyone_launcher.py`
- WebUI-App auf Phase 4 angehoben und um eine lokale Launch-Route erweitert:
  - `src/owli_train/webui/app.py`
- Reader-/View-Model-Layer um FiftyOne-Launch-Targets fuer Dataset- und Eval-Details erweitert:
  - `src/owli_train/webui/readers.py`
  - `src/owli_train/webui/models.py`
- Dataset- und Eval-Detailseiten um kleine FiftyOne-Startpunkte und Fehlhinweise erweitert:
  - `src/owli_train/webui/templates/dataset_detail.html`
  - `src/owli_train/webui/templates/eval_detail.html`
  - `src/owli_train/webui/templates/fiftyone_launch.html`
  - `src/owli_train/webui/templates/base.html`
- Kleine Tests fuer Reader, Route und fehlende FiftyOne-Abhaengigkeit ergaenzt:
  - `tests/test_webui_reader.py`
  - `tests/test_webui_app.py`
  - `tests/test_webui_fiftyone.py`
  - `tests/webui_test_utils.py`
- Optionalen Installationspfad fuer FiftyOne dokumentiert:
  - `requirements/fiftyone.txt`
  - `pyproject.toml`
- Doku auf den kleinen Phase-4-Scope aktualisiert:
  - `README.md`
  - `docs/webui.md`
  - `docs/runbook.md`

## Was wurde wirklich verifiziert?
- Statisch geprueft:
  - `README.md`
  - `docs/webui.md`
  - `docs/runbook.md`
  - `docs/review-templates/Codex-Task-Report.md`
  - `docs/reviews/Codex-Task-Report_last.md` (vorheriger Stand)
  - `src/owli_train/webui/*`
  - relevante COCO-/Eval-Pfadnutzung in:
    - `src/owli_train/data/merge_coco.py`
    - `src/owli_train/eval/detect.py`
    - `src/owli_train/data/materialize_images.py`
- Inhaltlich verifiziert:
  - Dataset-Detailseiten markieren nur solche Datasets als FiftyOne-faehig, die einen COCO-Pfad plus lokales `<dataset>/images` haben.
  - Eval-Detailseiten markieren nur solche Reports als FiftyOne-faehig, die repo-lokale `coco_path`- und `images_dir`-Felder enthalten.
  - Fehlende Bilder, fehlende COCO-Pfade oder ungeeignete Artefakte fuehren zu klaren UI-Hinweisen statt zu WebUI-Startfehlern.
  - Fehlendes `fiftyone` fuehrt nur auf der Launch-Route zu einer klaren Fehlermeldung; der normale WebUI-Start bleibt funktionsfaehig.
- Real ausgefuehrt:
  - gezielte WebUI-/FiftyOne-Tests
  - `ruff format`, `ruff check`, kompletter `pytest`-Lauf
  - lokaler Uvicorn-Start gegen das echte Repo
  - echte HTTP-GETs auf Dashboard-/Artifacts-/Jobs-Seiten gegen das echte Repo
  - lokaler Uvicorn-Start gegen ein temporaeres Sample-Repo mit Mini-Artefakten
  - echte HTTP-GETs auf Dataset-/Eval-/FiftyOne-Routen gegen das Sample-Repo
  - lokale Pruefung, dass `fiftyone` in der aktuellen venv nicht installiert ist
- Nicht real verifiziert:
  - ein echter erfolgreicher FiftyOne-App-Start, weil `fiftyone` in der aktuellen lokalen venv nicht installiert ist
  - die Runtime-Interaktion von `src/owli_train/webui/fiftyone_launcher.py` mit einer echten FiftyOne-Installation

## Tests
- `python - <<'PY'\nimport importlib.util\nprint(importlib.util.find_spec('fiftyone'))\nPY`
  - Exit-Code: `0`
  - Ergebnis: `None`
- `python -m pytest tests/test_webui_reader.py tests/test_webui_app.py tests/test_webui_fiftyone.py`
  - Exit-Code: `0`
  - Ergebnis: `11 passed in 0.66s`
- `python -m ruff check src/owli_train/webui tests/test_webui_app.py tests/test_webui_reader.py tests/test_webui_fiftyone.py tests/webui_test_utils.py`
  - Exit-Code: `0`
  - Ergebnis: `All checks passed!`
- `python -m ruff format .`
  - Exit-Code: `0`
  - Ergebnis: `2 files reformatted, 79 files left unchanged`
- `python -m ruff check .`
  - Exit-Code: `0`
  - Ergebnis: `All checks passed!`
- `python -m pytest`
  - Exit-Code: `0`
  - Ergebnis: `182 passed, 5 skipped in 4.84s`
- `PYTHONPATH=src python -m uvicorn owli_train.webui.app:app --host 127.0.0.1 --port 8000`
  - Exit-Code: `0` nach sauberem `CTRL+C`
  - Ergebnis: WebUI startete lokal gegen das echte Repo
- `python - <<'PY'\nimport httpx\n\nbase = 'http://127.0.0.1:8000'\npaths = ['/', '/artifacts', '/jobs']\nwith httpx.Client(base_url=base, timeout=10.0) as client:\n    for path in paths:\n        response = client.get(path)\n        print(path, response.status_code, len(response.text))\nPY`
  - Exit-Code: `0`
  - Ergebnis:
    - `/` -> `200`
    - `/artifacts` -> `200`
    - `/jobs` -> `200`
- `PYTHONPATH=src:. python - <<'PY'\nimport tempfile\nfrom pathlib import Path\n\nimport uvicorn\n\nfrom owli_train.webui.app import create_app\nfrom tests.webui_test_utils import build_sample_repo\n\nrepo_root = build_sample_repo(Path(tempfile.mkdtemp(prefix='owli-webui-sample-')))\nprint(repo_root)\nuvicorn.run(create_app(repo_root=repo_root), host='127.0.0.1', port=8001)\nPY`
  - Exit-Code: `0` nach sauberem `CTRL+C`
  - Ergebnis: WebUI startete lokal gegen ein temporaeres Sample-Repo mit geeignetem Mini-Dataset
- `python - <<'PY'\nimport httpx\n\nbase = 'http://127.0.0.1:8001'\npaths = [\n    '/',\n    '/datasets/view?path=work/datasets/demo-dataset',\n    '/evals/view?path=work/runs/20260309-123000-demo/reports/eval_demo.json',\n    '/fiftyone/open?source=dataset&path=work/datasets/demo-dataset',\n]\nwith httpx.Client(base_url=base, timeout=10.0) as client:\n    for path in paths:\n        response = client.get(path)\n        print(path, response.status_code, len(response.text))\n        if path.startswith('/fiftyone/open'):\n            marker = 'FiftyOne is not installed in this venv.' in response.text\n            print('missing_dependency_message', marker)\nPY`
  - Exit-Code: `0`
  - Ergebnis:
    - `/` -> `200`
    - `/datasets/view?...` -> `200`
    - `/evals/view?...` -> `200`
    - `/fiftyone/open?...` -> `503`
    - `missing_dependency_message` -> `True`

## Relevante Run-Kommandos
- WebUI lokal in WSL2 starten:
```bash
source .venv/bin/activate
PYTHONPATH=src python -m uvicorn owli_train.webui.app:app --host 127.0.0.1 --port 8000 --reload
```
- Optionalen FiftyOne-Pfad im selben WebUI-venv installieren:
```bash
source .venv/bin/activate
pip install -r requirements/fiftyone.txt
```
- PowerShell-Aequivalent fuer FiftyOne:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements\fiftyone.txt
```

## Offene Risiken
- Der erfolgreiche Runtime-Pfad in `src/owli_train/webui/fiftyone_launcher.py` konnte ohne lokale FiftyOne-Installation nur statisch geprueft werden.
- Der aktuelle Scope oeffnet nur kuratierte COCO-Datasets mit klar aufloesbarem Images-Root; andere Repo-Artefakte werden absichtlich nicht heuristisch erraten.
- Der Service verwaltet bewusst nur einen kleinen lokalen Launch-Pfad und keine robuste Mehrfach- oder Mehrbenutzer-Session-Verwaltung.

## Nächster sinnvoller Schritt
- Ergaenze auf der Run-Detailseite je Eval-Report einen direkten FiftyOne-Kurzlink zum dort referenzierten Dataset, damit der Umweg ueber die Eval-Detailseite fuer den haeufigsten Analysepfad entfaellt.
