# Codex Task Report

## Ziel
- Die bestehende WebUI-Compare-Seite um kleine Delta-Spalten gegen einen Baseline-Run erweitern.
- Neben den vorhandenen globalen und kuratierten Per-Class-Metriken auch die relative Einordnung gegen eine Referenz direkt im Browser sichtbar machen.
- Den Ausbau bewusst klein halten: nur einfache Differenzen, keine neue Analyseplattform und keine neue Persistenz.

## Was wurde geändert?
- Compare-View-Modelle um Baseline- und Delta-Felder erweitert:
  - Baseline-Optionen fuer `/compare/runs`
  - globale Delta-Metriken pro Zeile
  - kuratierte Per-Class-Delta-Metriken pro Zelle
  - Baseline-Markierung fuer globale und Per-Class-Zeilen
  - Datei: `src/owli_train/webui/models.py`
- Reader-Layer minimal um Baseline- und Delta-Aufbereitung erweitert:
  - Baseline-Regel: explizit gewaehlter Baseline-Run oder sonst erster selektierter Run oder sonst erste angezeigte Zeile
  - Deltas nur bei numerischen Werten auf beiden Seiten
  - einfache rohe Differenzen fuer `AP`, `AP50`, `AP75`, `AR100`, `precision`, `recall`
  - einfache rohe Differenzen fuer kuratierte Per-Class-Metriken `precision`, `recall`, `tp`, `fp`, `fn`
  - defensive Alias-Nutzung bleibt unveraendert nur innerhalb kuratierter Klassenzeilen
  - Datei: `src/owli_train/webui/readers.py`
- Compare-Route um den kleinen Query-Parameter `baseline` erweitert:
  - Datei: `src/owli_train/webui/app.py`
- Compare-Template um Baseline-Auswahl, Baseline-Hinweis, Delta-Spalten und leichte Baseline-Hervorhebung erweitert:
  - Datei: `src/owli_train/webui/templates/compare_runs.html`
  - Datei: `src/owli_train/webui/templates/base.html`
- Tests gezielt fuer Delta-Berechnung, Baseline-Standardregel, manuelle Baseline-Auswahl und Rendern der Delta-Sicht erweitert:
  - `tests/test_webui_reader.py`
  - `tests/test_webui_app.py`
- Doku auf den kleinen Delta-Scope aktualisiert:
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
- Inhaltlich verifiziert:
  - die bisherige globale Compare-Tabelle bleibt erhalten und zeigt zusaetzlich Delta-Spalten
  - die Baseline-Auswahl bleibt klein und nutzt nur einen einzelnen Referenz-Run
  - Delta-Werte werden nur bei numerisch vorhandenen Werten auf beiden Seiten berechnet
  - fehlende globale oder Per-Class-Werte bleiben als `-` sichtbar
  - Baseline-Zeilen und -Zellen werden klar markiert und zeigen `baseline` statt einer Differenz
  - die bestehende defensive Alias-Behandlung fuer `obstacle_fence` / `obstacle_fence_rail` und `obstacle_hole` / `obstacle_hole_dropoff` bleibt erhalten
- Real ausgefuehrt:
  - gezielte WebUI-Reader- und Route-Tests fuer Delta-Logik und Baseline-Auswahl
  - `ruff format`, `ruff check`, kompletter `pytest`-Lauf
  - lokaler Uvicorn-Start gegen das echte Repo
  - echte HTTP-GETs auf `/compare/runs` gegen das echte Repo
  - lokaler Uvicorn-Start gegen ein temporaeres Sample-Repo mit mehreren Compare-Runs
  - echte HTTP-GETs auf `/compare/runs` gegen das Sample-Repo fuer:
    - Default-Baseline
    - explizite Baseline-Auswahl
    - BA-core-plus-rehearsal-Sicht mit Delta-Spalten
- Nicht real verifiziert:
  - eine Multi-Run-Delta-Ansicht gegen echte lokale Repo-Artefakte unter `work/runs`, weil dieses Verzeichnis im aktuellen Repo-Workspace weiterhin fehlt
  - eine komplexere Mehrfach-Baseline-Logik, weil sie bewusst nicht Teil des Scopes ist

## Tests
- `python -m pytest tests/test_webui_reader.py tests/test_webui_app.py`
  - Exit-Code: `0`
  - Ergebnis: `20 passed in 2.87s`
- `python -m pytest tests/test_webui_app.py`
  - Exit-Code: `0`
  - Ergebnis: `8 passed in 1.81s`
- `python -m ruff format .`
  - Exit-Code: `0`
  - Ergebnis: `2 files reformatted, 79 files left unchanged`
- `python -m ruff check .`
  - Exit-Code: `0`
  - Ergebnis: `All checks passed!`
- `python -m pytest`
  - Exit-Code: `0`
  - Ergebnis: `195 passed, 5 skipped in 7.81s`
- `PYTHONPATH=src python -m uvicorn owli_train.webui.app:app --host 127.0.0.1 --port 8000`
  - Exit-Code: `0` nach sauberem `CTRL+C`
  - Ergebnis: WebUI startete lokal gegen das echte Repo
- `python - <<'PY'\nimport httpx\n\nbase = 'http://127.0.0.1:8000'\npaths = ['/compare/runs']\nwith httpx.Client(base_url=base, timeout=10.0) as client:\n    for path in paths:\n        response = client.get(path)\n        print(path, response.status_code, len(response.text))\n        print('compare_title', 'Run / Eval Compare' in response.text)\n        print('delta_header', 'Delta AP50' in response.text)\n        print('baseline_note', 'Baseline reference:' in response.text)\nPY`
  - Exit-Code: `0`
  - Ergebnis:
    - `/compare/runs` -> `200`
    - `compare_title=True`
    - `delta_header=False`
    - `baseline_note=False`
    - Grund: Im aktuellen echten Repo fehlen weiter `work/runs`-Artefakte, daher rendert die leere Compare-Seite ohne Datenzeilen und ohne Delta-Header
- `PYTHONPATH=src:. python - <<'PY'\nimport tempfile\nfrom pathlib import Path\n\nimport uvicorn\n\nfrom owli_train.webui.app import create_app\nfrom tests.webui_test_utils import build_sample_repo\n\nrepo_root = build_sample_repo(Path(tempfile.mkdtemp(prefix='owli-webui-delta-')))\nprint(repo_root)\nuvicorn.run(create_app(repo_root=repo_root), host='127.0.0.1', port=8001)\nPY`
  - Exit-Code: `0` nach sauberem `CTRL+C`
  - Ergebnis: WebUI startete lokal gegen ein temporaeres Sample-Repo mit mehreren Compare-Runs
- `python - <<'PY'\nimport httpx\n\nbase = 'http://127.0.0.1:8001'\npaths = [\n    '/compare/runs',\n    '/compare/runs?baseline=work/runs/20260308-211806-ba-mvp-stage4-20260308',\n    '/compare/runs?class_scope=ba_core_rehearsal',\n]\nwith httpx.Client(base_url=base, timeout=10.0) as client:\n    for path in paths:\n        response = client.get(path)\n        print(path, response.status_code, len(response.text))\n        print('delta_header', 'Delta AP50' in response.text)\n        print('baseline_note', 'Baseline reference:' in response.text)\n        print('delta_value', '+0.0035' in response.text or '-0.0035' in response.text)\n        print('per_class_delta', '<code>tp</code>: baseline' in response.text)\nPY`
  - Exit-Code: `0`
  - Ergebnis:
    - Default-Compare rendert mit Delta-Spalten
    - explizite Baseline-Auswahl rendert mit angepasster Referenz
    - BA-core-plus-rehearsal-Sicht rendert mit Per-Class-Delta-Zellen

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
- Optional eine explizite Baseline waehlen:
```text
http://127.0.0.1:8000/compare/runs?baseline=work/runs/<run-id>
```
- Optional die kuratierte Per-Class-Sicht mit Rehearsal-Klassen aktivieren:
```text
http://127.0.0.1:8000/compare/runs?class_scope=ba_core_rehearsal
```

## Offene Risiken
- Die Delta-Sicht bleibt bewusst bei einfachen rohen Differenzen; sie bewertet nicht automatisch, ob ein positives oder negatives Vorzeichen fachlich besser ist.
- Pro Compare-Ansicht wird bewusst nur eine Baseline gleichzeitig unterstuetzt.
- Im echten Repo-Workspace konnte wegen fehlender `work/runs`-Artefakte nur das Rendern der leeren Compare-Seite live gegen das aktuelle Repo verifiziert werden; die Multi-Run-Delta-Ansicht wurde live gegen das Test-Sample verifiziert.

## Nächster sinnvoller Schritt
- Ergaenze optional auf `/compare/runs` eine kleine feste Sortierung nach einem ausgewaehlten Delta-Feld, ohne den UI-Scope auf eine generische Benchmark-Plattform auszuweiten.
