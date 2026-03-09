# Codex Task Report

## Ziel
- Die bestehende read-only WebUI um eine kleine, sichere Job-Steuerung erweitern.
- Nur eine kleine Whitelist leichter CLI-Kommandos direkt aus dem Browser startbar machen.
- Jobstatus, Exit-Code, Logs und erwartete Artefaktpfade persistent sichtbar machen, ohne die CLI zu ersetzen oder eine grosse Queue-/Worker-Plattform einzufuehren.

## Was wurde geaendert?
- WebUI-App um Jobseiten und Launcher-Routen erweitert:
  - `src/owli_train/webui/app.py`
- Neuer kleiner file-backed Job-Layer mit Whitelist, Store, Launcher und Worker:
  - `src/owli_train/webui/jobs.py`
  - `src/owli_train/webui/worker.py`
- Neue Job-Templates ergaenzt:
  - `src/owli_train/webui/templates/jobs.html`
  - `src/owli_train/webui/templates/job_detail.html`
- Bestehende WebUI-Modelle, Reader und Basis-Templates fuer Jobanzeige erweitert:
  - `src/owli_train/webui/models.py`
  - `src/owli_train/webui/readers.py`
  - `src/owli_train/webui/templates/base.html`
  - `src/owli_train/webui/templates/dashboard.html`
- Kleine Tests fuer Job-Store, Worker, Launcher-Routen und Whitelist-Command-Building ergaenzt:
  - `tests/test_webui_jobs.py`
  - `tests/test_webui_app.py`
  - `tests/webui_test_utils.py`
- Doku und WebUI-Abhaengigkeiten auf Phase 2 aktualisiert:
  - `README.md`
  - `docs/runbook.md`
  - `docs/webui.md`
  - `requirements/webui.txt`
  - `pyproject.toml`

## Was wurde wirklich verifiziert?
- Statisch geprueft:
  - `README.md`
  - `docs/runbook.md`
  - `docs/webui.md`
  - `src/owli_train/webui/*`
  - `src/owli_train/cli.py`
  - relevante Datenpfade:
    - `src/owli_train/data/coco.py`
    - `src/owli_train/data/split.py`
    - `src/owli_train/data/merge_coco.py`
    - `src/owli_train/data/materialize_images.py`
    - `src/owli_train/data/modelmaker_csv.py`
  - `docs/reviews/Codex-Task-Report_last.md` (vorheriger Stand)
- Inhaltlich verifiziert:
  - der kleinste robuste lokale Job-Mechanismus ist ein file-backed JSON-Store pro Job plus eigener Python-Worker-Subprozess
  - die Phase-2-Whitelist bleibt bewusst klein:
    - `dataset validate`
    - `dataset split`
    - `dataset merge coco`
    - `dataset export modelmaker-csv`
    - `dataset materialize-images` mit Merge-Manifest
  - keine freie Shell-Eingabe wird zugelassen
  - Inputs fuer bestehende Dateien kommen nur aus bekannten Repo-Pfaden; Outputs werden auf `work/` begrenzt
- Real ausgefuehrt:
  - WebUI-Abhaengigkeiten installiert
  - Ruff Format / Ruff Check / Pytest ausgefuehrt
  - lokaler Uvicorn-Start der App
  - echter HTTP-GET auf `/jobs`
  - echter HTTP-POST auf `/jobs/launch/dataset_split`
  - echter persistenter Split-Job ueber die neue Jobs-Route bis `succeeded` verifiziert
  - echte Job-Detailseite per HTTP geladen

## Tests
- `python -m pip install -r requirements/dev.txt`
  - Exit-Code: `0`
- `python -m ruff format .`
  - Exit-Code: `0`
  - Ergebnis: `2 files reformatted, 76 files left unchanged`
- `python -m ruff check .`
  - Exit-Code: `1`
  - Ergebnis: kleine Import-/Typing-Korrekturen in `src/owli_train/webui/jobs.py`
- `python -m ruff check src/owli_train/webui/jobs.py --fix`
  - Exit-Code: `0`
  - Ergebnis: `3` kleine Ruff-Fixes automatisch angewendet
- `python -m ruff check .`
  - Exit-Code: `0`
  - Ergebnis: `All checks passed!`
- `python -m pytest`
  - Exit-Code: `2`
  - Ergebnis: erster Lauf scheiterte an Python-3.10-Inkompatibilitaet durch `datetime.UTC`
- `python -m ruff check src/owli_train/webui/app.py`
  - Exit-Code: `0`
- `python -m pytest`
  - Exit-Code: `0`
  - Ergebnis: `170 passed, 5 skipped in 5.37s`
- `python -m ruff format .`
  - Exit-Code: `0`
  - Ergebnis: `1 file reformatted, 77 files left unchanged`
- `python -m ruff check .`
  - Exit-Code: `0`
  - Ergebnis: `All checks passed!`
- `python -m pytest`
  - Exit-Code: `0`
  - Ergebnis: `171 passed, 5 skipped in 5.67s`
- `PYTHONPATH=src python -m uvicorn owli_train.webui.app:app --host 127.0.0.1 --port 8000`
  - Exit-Code: `0` nach sauberem `CTRL+C`
  - Ergebnis: lokale App auf `http://127.0.0.1:8000`
- 
  ```bash
  python - <<'PY'
  import time
  from pathlib import Path

  import httpx

  from owli_train.webui.jobs import JobService

  base = 'http://127.0.0.1:8000'
  repo_root = Path('/home/michael/src/2/owli-detector-train-keras')
  service = JobService(repo_root)

  with httpx.Client(base_url=base, timeout=10.0, follow_redirects=False) as client:
      jobs_page = client.get('/jobs')
      print('GET /jobs', jobs_page.status_code)
      launch = client.post(
          '/jobs/launch/dataset_split',
          data={
              'coco_path': 'tests/data/coco_min.json',
              'out_dir': 'work/webui/splits/verification',
              'seed': '1337',
              'write_coco': 'on',
              'ensure_train_class_coverage': 'on',
          },
      )
      print('POST /jobs/launch/dataset_split', launch.status_code)
      location = launch.headers.get('location', '')
      print('Location', location)

  job_id = location.rstrip('/').split('/')[-1]
  for _ in range(60):
      detail = service.get_job_detail(job_id)
      if detail is not None and detail.record.status in {'succeeded', 'failed'}:
          print('Job status', detail.record.status)
          print('Exit code', detail.record.exit_code)
          print('Artifacts present', [artifact.exists for artifact in detail.artifact_views])
          break
      time.sleep(0.1)
  PY
  ```
  - Exit-Code: `0`
  - Ergebnis:
    - `GET /jobs 200`
    - `POST /jobs/launch/dataset_split 303`
    - realer Job endete mit `succeeded`
    - Exit-Code `0`
    - erwartete Split-Artefakte vorhanden
- 
  ```bash
  python - <<'PY'
  import httpx
  from pathlib import Path
  from owli_train.webui.jobs import JobService

  service = JobService(Path('/home/michael/src/2/owli-detector-train-keras'))
  job_id = service.list_jobs(limit=1)[0].job_id
  with httpx.Client(base_url='http://127.0.0.1:8000', timeout=10.0) as client:
      response = client.get(f'/jobs/{job_id}')
      print(response.status_code)
      print('dataset split' in response.text)
      print('succeeded' in response.text)
  PY
  ```
  - Exit-Code: `0`
  - Ergebnis: Job-Detailseite antwortete mit `200`, Typ und Status waren im HTML sichtbar

## Relevante Run-Kommandos
- WSL2-Start der WebUI:
```bash
source .venv/bin/activate
PYTHONPATH=src python -m uvicorn owli_train.webui.app:app --host 127.0.0.1 --port 8000 --reload
```
- Beispiel fuer einen realen UI-Job via HTTP:
```bash
python - <<'PY'
import httpx
with httpx.Client(base_url='http://127.0.0.1:8000', follow_redirects=False) as client:
    response = client.post(
        '/jobs/launch/dataset_split',
        data={
            'coco_path': 'tests/data/coco_min.json',
            'out_dir': 'work/webui/splits/verification',
            'seed': '1337',
            'write_coco': 'on',
            'ensure_train_class_coverage': 'on',
        },
    )
    print(response.status_code, response.headers.get('location'))
PY
```

## Offene Risiken
- Die Phase-2-Jobs bleiben bewusst lokal und klein; es gibt noch keine Abbruch-/Cancel-Funktion fuer laufende Jobs.
- Die sichere Input-Auswahl basiert auf bekannten Repo-Pfaden; neue oder ungewoehnliche Datenpfade tauchen erst auf, wenn sie in den Auswahlkatalogen erfasst sind.
- `dataset materialize-images` ist absichtlich nur ueber Merge-Manifeste freigeschaltet; der UI-Pfad deckt nicht die freie CLI-Variante mit beliebigen `--source-images-dir`-Listen ab.

## Naechster sinnvoller Schritt
- Ergaenze als naechsten kleinen Schritt eine staerkere Dataset-Analyse-/Visualisierung in der WebUI, zum Beispiel ueber interne Dataset-Statistikseiten oder eine erste eng begrenzte FiftyOne-Anbindung.
