# Codex Task Report

## Ziel
- Die bestehende WebUI-Compare-Seite um kleine, kuratierte Per-Class-Vergleiche erweitern.
- Neben globalen Metriken auch die wichtigsten BA-Core- und ausgewaehlten Rehearsal-Klassen direkt im Browser vergleichbar machen.
- Den Ausbau bewusst klein halten: bestehende Eval-JSONs nutzen, keine neue Analyseplattform, keine neue Persistenz.

## Was wurde geändert?
- Compare-View-Modelle um kuratierte Per-Class-Daten erweitert:
  - Scope-Auswahl fuer `BA core only` und `BA core + rehearsal`
  - Per-Class-Zellen und -Zeilen fuer die Compare-Seite
  - Datei: `src/owli_train/webui/models.py`
- Reader-Layer minimal um Per-Class-Compare-Aufbereitung erweitert:
  - liest `per_class` direkt aus bestehenden `eval*.json`-Reports
  - extrahiert `precision`, `recall`, `tp`, `fp`, `fn`
  - fuehrt nur defensive Alias-Gruppierung innerhalb einzelner kuratierter Zeilen aus:
    - `obstacle_fence` / `obstacle_fence_rail`
    - `obstacle_hole` / `obstacle_hole_dropoff`
  - haelt `obstacle_ground` und `obstacle_barrier` getrennt statt sie heuristisch zusammenzuwerfen
  - rendert nur solche kuratierten Klassenzeilen, fuer die mindestens ein selektierter Eval-Report Daten liefert
  - Datei: `src/owli_train/webui/readers.py`
- Compare-Route um den kleinen Query-Parameter `class_scope` erweitert:
  - Datei: `src/owli_train/webui/app.py`
- Compare-Template um eine zweite kompakte Per-Class-Tabelle erweitert:
  - globale Vergleichstabelle bleibt bestehen
  - darunter kuratierte Per-Class-Sicht mit Scope-Auswahl und klaren `-`-Werten bei fehlenden Klassen
  - Datei: `src/owli_train/webui/templates/compare_runs.html`
- Kleine Phase-6-Textanpassungen in Navigation und Dashboard:
  - `src/owli_train/webui/templates/base.html`
  - `src/owli_train/webui/templates/dashboard.html`
- Fixtures und Tests gezielt fuer Alias-Namen, fehlende Klassen und Scope-Umschalter erweitert:
  - `tests/webui_test_utils.py`
  - `tests/test_webui_reader.py`
  - `tests/test_webui_app.py`
- Doku auf den kleinen Phase-6-Scope aktualisiert:
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
  - vorhandene Eval-Report-Struktur in `src/owli_train/eval/efficientdet_tflite.py`
- Inhaltlich verifiziert:
  - die globale Compare-Tabelle bleibt unveraendert erhalten
  - die neue Per-Class-Sicht nutzt nur die vorhandenen `per_class`-Felder aus den Eval-JSONs
  - historische Alias-Namen werden nur innerhalb kuratierter Klassenzeilen gematcht
  - fehlende Klassen oder fehlende `per_class`-Bloecke fuehren zu klaren `-`-Werten oder zu einer leeren Per-Class-Sektion statt zu Fehlern
  - der Scope-Umschalter `BA core only` vs `BA core + rehearsal` arbeitet ueber denselben Compare-Pfad
- Real ausgefuehrt:
  - gezielte WebUI-Reader- und Route-Tests fuer die Per-Class-Sicht
  - `ruff format`, `ruff check`, kompletter `pytest`-Lauf
  - lokaler Uvicorn-Start gegen das echte Repo
  - echte HTTP-GETs auf `/` und `/compare/runs` gegen das echte Repo
  - lokaler Uvicorn-Start gegen ein temporaeres Sample-Repo mit mehreren Compare-Runs und Per-Class-Daten
  - echte HTTP-GETs auf `/compare/runs` gegen das Sample-Repo fuer:
    - Default-Per-Class-Sicht
    - `BA core + rehearsal`
    - gefilterte Stage-4-Ansicht mit `class_scope`
- Nicht real verifiziert:
  - eine Multi-Run-Per-Class-Ansicht gegen echte lokale Repo-Artefakte unter `work/runs`, weil dieses Verzeichnis im aktuellen Repo-Workspace weiterhin fehlt
  - eine vollstaendige beliebige Per-Class-Auswahl, weil diese bewusst nicht Teil des kleinen Scopes ist

## Tests
- `python -m pytest tests/test_webui_reader.py tests/test_webui_app.py`
  - Exit-Code: `0`
  - Ergebnis: `16 passed in 2.66s`
- `python -m ruff format .`
  - Exit-Code: `0`
  - Ergebnis: `2 files reformatted, 79 files left unchanged`
- `python -m ruff check .`
  - Exit-Code: `0`
  - Ergebnis: `All checks passed!`
- `python -m pytest`
  - Exit-Code: `0`
  - Ergebnis: `188 passed, 5 skipped in 7.74s`
- `PYTHONPATH=src python -m uvicorn owli_train.webui.app:app --host 127.0.0.1 --port 8000`
  - Exit-Code: `0` nach sauberem `CTRL+C`
  - Ergebnis: WebUI startete lokal gegen das echte Repo
- `python - <<'PY'\nimport httpx\n\nbase = 'http://127.0.0.1:8000'\npaths = ['/', '/compare/runs']\nwith httpx.Client(base_url=base, timeout=10.0) as client:\n    for path in paths:\n        response = client.get(path)\n        print(path, response.status_code, len(response.text))\n        if path == '/compare/runs':\n            print('compare_title', 'Run / Eval Compare' in response.text)\n            print('per_class_title', 'Curated per-class view' in response.text)\nPY`
  - Exit-Code: `0`
  - Ergebnis:
    - `/` -> `200`
    - `/compare/runs` -> `200`
    - Compare- und Per-Class-Ueberschrift im HTML vorhanden
- `PYTHONPATH=src:. python - <<'PY'\nimport tempfile\nfrom pathlib import Path\n\nimport uvicorn\n\nfrom owli_train.webui.app import create_app\nfrom tests.webui_test_utils import build_sample_repo\n\nrepo_root = build_sample_repo(Path(tempfile.mkdtemp(prefix='owli-webui-per-class-')))\nprint(repo_root)\nuvicorn.run(create_app(repo_root=repo_root), host='127.0.0.1', port=8001)\nPY`
  - Exit-Code: `0` nach sauberem `CTRL+C`
  - Ergebnis: WebUI startete lokal gegen ein temporaeres Sample-Repo mit Per-Class-Compare-Daten
- `python - <<'PY'\nimport httpx\n\nbase = 'http://127.0.0.1:8001'\npaths = [\n    '/compare/runs',\n    '/compare/runs?class_scope=ba_core_rehearsal',\n    '/compare/runs?run=work/runs/20260308-211806-ba-mvp-stage4-20260308&target=split:ba_mvp_stage4_with_coco_replay:test&class_scope=ba_core_rehearsal',\n]\nwith httpx.Client(base_url=base, timeout=10.0) as client:\n    for path in paths:\n        response = client.get(path)\n        print(path, response.status_code, len(response.text))\n        print('per_class_title', 'Curated per-class view' in response.text)\n        print('fence_alias', 'obstacle_fence_rail' in response.text)\n        print('rehearsal_scope', 'BA core + rehearsal' in response.text)\n        print('truck_row', 'truck' in response.text)\n        print('selection_summary', 'Showing 1 eval rows across 1 runs' in response.text)\nPY`
  - Exit-Code: `0`
  - Ergebnis:
    - Default-Compare rendert mit Per-Class-Sicht und Alias-Hinweis
    - `BA core + rehearsal` rendert mit zusaetzlichen Rehearsal-Zeilen wie `truck`
    - gefilterte Stage-4-Ansicht rendert mit `200` und passender Auswahlzusammenfassung

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
- Optional die Per-Class-Sicht auf BA-Core plus Rehearsal erweitern:
```text
http://127.0.0.1:8000/compare/runs?class_scope=ba_core_rehearsal
```
- Optional eine gefilterte Compare-Ansicht mit Ziel und Scope aufrufen:
```text
http://127.0.0.1:8000/compare/runs?run=work/runs/<run-id>&target=<target-key>&class_scope=ba_core_rehearsal
```

## Offene Risiken
- Die Per-Class-Sicht bleibt bewusst klein und kuratiert; sie ist keine generische Contract- oder Klassen-Explorer-Oberflaeche.
- Historische Alias-Namen werden nur innerhalb einzelner kuratierter Reihen defensiv zusammengefuehrt; weitergehende Semantik-Mappings zwischen BA-v1 und BA-v2 werden absichtlich nicht erfunden.
- Im echten Repo-Workspace konnte wegen fehlender `work/runs`-Artefakte nur das Rendern der leeren Compare-/Per-Class-Seite live gegen das aktuelle Repo verifiziert werden; die Multi-Run-Per-Class-Ansicht wurde live gegen das Test-Sample verifiziert.

## Nächster sinnvoller Schritt
- Ergaenze optional auf `/compare/runs` eine kleine, feste Delta-Spalte gegen den zuerst selektierten Baseline-Run fuer die kuratierten Per-Class-Metriken, ohne den UI-Scope auf freie Benchmark-Analysen auszuweiten.
