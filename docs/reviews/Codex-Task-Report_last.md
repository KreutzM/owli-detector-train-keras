# Codex Task Report

## Ziel
- Die bestehende FastAPI-WebUI um eine kleine, robuste Run-/Eval-Vergleichsseite erweitern.
- Mehrere relevante Baselines und Experimentlaeufe im Browser ueber ihre wichtigsten Eval-Metriken vergleichbar machen.
- Den Scope bewusst klein halten: bestehende Reader nutzen, keine neue Datenbank, keine Benchmark-Plattform, keine neue Job-Steuerung.

## Was wurde geändert?
- Compare-View-Modelle fuer Run-Auswahl, Eval-Zielgruppen und Vergleichszeilen ergaenzt:
  - `src/owli_train/webui/models.py`
- Reader-Layer um eine kleine Compare-Aufbereitung erweitert:
  - scannt strukturierte `eval*.json`-Reports unter `work/runs/*/reports/`
  - gruppiert vergleichbare Reports ueber gemeinsame Eval-Ziele
  - extrahiert `AP`, `AP50`, `AP75`, `AR100`, `precision`, `recall`
  - matched Run-Config-Snapshots wenn moeglich auf eingecheckte `configs/*.yaml`
  - behandelt fehlende Metriken oder partielle Artefakte mit `-` statt Abbruch
  - Dateien: `src/owli_train/webui/readers.py`
- Neue WebUI-Route und Template fuer die Vergleichsansicht ergaenzt:
  - Route: `/compare/runs`
  - Template: `src/owli_train/webui/templates/compare_runs.html`
  - Datei: `src/owli_train/webui/app.py`
- Navigation und Ruecklinks auf die neue Vergleichsansicht erweitert:
  - `src/owli_train/webui/templates/base.html`
  - `src/owli_train/webui/templates/dashboard.html`
  - `src/owli_train/webui/templates/run_detail.html`
  - `src/owli_train/webui/templates/eval_detail.html`
  - `src/owli_train/webui/templates/golden_detail.html`
- Kleine Robustheitsanpassung an bestehende Run-Dateinamenschemata:
  - `config.yaml` und `mapping_files.json` werden jetzt neben den aelteren Snapshot-Namen erkannt
  - Datei: `src/owli_train/webui/readers.py`
- Gezielt erweiterte WebUI-Tests inkl. Compare-Fixture mit mehreren Baselines:
  - `tests/webui_test_utils.py`
  - `tests/test_webui_reader.py`
  - `tests/test_webui_app.py`
- Doku auf den kleinen Compare-Scope aktualisiert:
  - `README.md`
  - `docs/webui.md`
  - `docs/runbook.md`

## Was wurde wirklich verifiziert?
- Statisch geprueft:
  - `README.md`
  - `docs/webui.md`
  - `docs/runbook.md`
  - `docs/review-templates/Codex-Task-Report.md`
  - bestehende WebUI-Dateien unter `src/owli_train/webui/`
  - Eval-Report-Struktur in `src/owli_train/eval/efficientdet_tflite.py`
  - Training-Run-Artefakte in `src/owli_train/training/modelmaker_efficientdet.py`
- Inhaltlich verifiziert:
  - die Compare-Seite liest nur strukturierte `eval*.json`-Artefakte, keine Markdown-Reports
  - die Default-Auswahl nimmt das Eval-Ziel mit der groessten Run-Abdeckung nach aktueller Run-Auswahl
  - bekannte Baselines werden nur ueber konkrete Namensmuster markiert, nicht ueber freie Vermutungen
  - fehlende Metrikfelder bleiben sichtbar als `-`
  - Run-/Eval-/Golden-Details bleiben aus der Compare-Seite direkt verlinkt
- Real ausgefuehrt:
  - gezielte WebUI-Reader- und Route-Tests
  - `ruff format`, `ruff check`, kompletter `pytest`-Lauf
  - lokaler Uvicorn-Start gegen das echte Repo
  - echte HTTP-GETs auf `/`, `/compare/runs`, `/jobs` gegen das echte Repo
  - lokaler Uvicorn-Start gegen ein temporaeres Sample-Repo mit mehreren Compare-Runs
  - echte HTTP-GETs auf `/compare/runs` gegen das Sample-Repo fuer Default- und gefilterte Compare-Ansichten
- Nicht real verifiziert:
  - ein Compare-Lauf mit echten lokalen Repo-Run-Artefakten unter `work/runs`, weil im aktuellen Repo-Workspace kein solches Verzeichnis vorhanden war
  - weitergehende per-class-Vergleichsansichten, weil diese bewusst nicht Teil dieses kleinen Scopes sind

## Tests
- `python -m pytest tests/test_webui_reader.py tests/test_webui_app.py`
  - Exit-Code: `0`
  - Ergebnis: `13 passed in 0.89s`
- `python -m ruff format .`
  - Exit-Code: `0`
  - Ergebnis: `2 files reformatted, 79 files left unchanged`
- `python -m ruff check .`
  - Exit-Code: `0`
  - Ergebnis: `All checks passed!`
- `python -m pytest`
  - Exit-Code: `0`
  - Ergebnis: `185 passed, 5 skipped in 7.62s`
- `PYTHONPATH=src python -m uvicorn owli_train.webui.app:app --host 127.0.0.1 --port 8000`
  - Exit-Code: `0` nach sauberem `CTRL+C`
  - Ergebnis: WebUI startete lokal gegen das echte Repo
- `python - <<'PY'\nimport httpx\n\nbase = 'http://127.0.0.1:8000'\npaths = ['/', '/compare/runs', '/jobs']\nwith httpx.Client(base_url=base, timeout=10.0) as client:\n    for path in paths:\n        response = client.get(path)\n        marker = 'Run / Eval Compare' in response.text if path == '/compare/runs' else True\n        print(path, response.status_code, len(response.text), marker)\nPY`
  - Exit-Code: `0`
  - Ergebnis:
    - `/` -> `200`
    - `/compare/runs` -> `200`
    - `/jobs` -> `200`
    - Compare-Marker im HTML vorhanden
- `PYTHONPATH=src:. python - <<'PY'\nimport tempfile\nfrom pathlib import Path\n\nimport uvicorn\n\nfrom owli_train.webui.app import create_app\nfrom tests.webui_test_utils import build_sample_repo\n\nrepo_root = build_sample_repo(Path(tempfile.mkdtemp(prefix='owli-webui-compare-')))\nprint(repo_root)\nuvicorn.run(create_app(repo_root=repo_root), host='127.0.0.1', port=8001)\nPY`
  - Exit-Code: `0` nach sauberem `CTRL+C`
  - Ergebnis: WebUI startete lokal gegen ein temporaeres Sample-Repo mit mehreren Baseline-/Compare-Runs
- `python - <<'PY'\nimport httpx\n\nbase = 'http://127.0.0.1:8001'\npaths = [\n    '/compare/runs',\n    '/compare/runs?run=work/runs/20260308-211806-ba-mvp-stage4-20260308&target=split:ba_mvp_stage4_with_coco_replay:test',\n]\nwith httpx.Client(base_url=base, timeout=10.0) as client:\n    for path in paths:\n        response = client.get(path)\n        print(path, response.status_code, len(response.text))\n        print('stage3_baseline', 'Stage-3 baseline' in response.text)\n        print('stage4_baseline', 'Stage-4 replay baseline' in response.text)\n        print('selection_summary', 'Showing 1 eval rows across 1 runs' in response.text)\nPY`
  - Exit-Code: `0`
  - Ergebnis:
    - Default-Compare rendert mit mehreren Baseline-Zeilen
    - gefilterte Stage-4-Compare rendert mit `200`
    - Filter-Zusammenfassung fuer `1` Run / `1` Eval-Zeile vorhanden

## Relevante Run-Kommandos
- WebUI lokal in WSL2 starten:
```bash
source .venv/bin/activate
PYTHONPATH=src python -m uvicorn owli_train.webui.app:app --host 127.0.0.1 --port 8000 --reload
```
- Compare-Seite lokal oeffnen:
```text
http://127.0.0.1:8000/compare/runs
```
- Optional eine gefilterte Compare-Ansicht ueber Query-Parameter aufrufen:
```text
http://127.0.0.1:8000/compare/runs?run=work/runs/<run-id>&target=<target-key>
```

## Offene Risiken
- Die erste Compare-Seite vergleicht bewusst nur globale Kennzahlen aus strukturierten Eval-JSONs; feinere per-class-Tabellen zwischen Runs sind noch nicht Teil des UI-Scopes.
- Die Config-Zuordnung ist absichtlich konservativ: nur exakte Snapshot-Matches auf eingecheckte `configs/*.yaml` werden als Repo-Config angezeigt.
- Im echten Repo-Workspace konnte mangels vorhandener `work/runs`-Artefakte nur das leere Rendern der Compare-Seite live gegen das aktuelle Repo verifiziert werden; die Multi-Run-Ansicht wurde live gegen das Test-Sample verifiziert.

## Nächster sinnvoller Schritt
- Ergaenze auf `/compare/runs` optional eine kleine, feste per-class-Zusatzspalte fuer wenige kuratierte BA-Core- und Rehearsal-Klassen, aber nur fuer das jeweils ausgewaehlte gemeinsame Eval-Ziel.
