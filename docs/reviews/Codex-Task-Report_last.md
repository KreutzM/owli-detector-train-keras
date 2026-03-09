# Codex Task Report

## Ziel
- Das erste schlanke WebUI-Grundgeruest fuer das Repo aufbauen.
- Eine kleine lokale FastAPI + Uvicorn App als read-only Einstiegspunkt ueber der bestehenden CLI-/Dateisystem-Pipeline liefern.
- Zunaechst nur stabile Repo-, Contract-, Dataset-, Run- und Artefaktinformationen im Browser sichtbar machen, ohne Job-Steuerung, DB oder Frontend-Overkill einzufuehren.

## Was wurde geaendert?
- Neuer read-only WebUI-Bereich unter `src/owli_train/webui/` ergaenzt:
  - `src/owli_train/webui/__init__.py`
  - `src/owli_train/webui/app.py`
  - `src/owli_train/webui/models.py`
  - `src/owli_train/webui/readers.py`
  - `src/owli_train/webui/templates/base.html`
  - `src/owli_train/webui/templates/dashboard.html`
  - `src/owli_train/webui/templates/contracts.html`
  - `src/owli_train/webui/templates/artifacts.html`
- Kleine read-only Reader-/View-Model-Schicht fuer:
  - Contract-Laden aus `configs/label_contracts/*.yaml`
  - kuratierte Doku-Referenzen
  - kuratierte Artefakt-Roots unter `work/`, `outputs`, `data`
  - erkannte Dataset-/Run-Verzeichnisse
  - config-basierte Pfadreferenzen aus `configs/*.yaml`
- Kleine Tests fuer Reader und FastAPI-Routen ergaenzt:
  - `tests/__init__.py`
  - `tests/webui_test_utils.py`
  - `tests/test_webui_reader.py`
  - `tests/test_webui_app.py`
- WebUI-Abhaengigkeiten und Repo-Doku minimal erweitert:
  - `requirements/webui.txt`
  - `requirements/dev.txt`
  - `pyproject.toml`
  - `README.md`
  - `docs/runbook.md`
  - `docs/webui.md`

## Was wurde wirklich verifiziert?
- Statisch geprueft:
  - `README.md`
  - `docs/runbook.md`
  - `docs/MVP_Training_Plan.md`
  - `docs/BA_v1_Labelset.md`
  - `docs/BA_v2_Hazard_Labelset.md`
  - `docs/reviews/Codex-Task-Report_last.md` (vorheriger Stand)
  - relevante Daten-/Trainings-/Eval-/Golden-Module unter:
    - `src/owli_train/data/*`
    - `src/owli_train/training/*`
    - `src/owli_train/eval/*`
    - `src/owli_train/golden/*`
  - relevante Configs unter:
    - `configs/label_contracts/*`
    - `configs/label_maps/*`
    - `configs/*.yaml`
- Inhaltlich verifiziert:
  - ein kleiner read-only Dateisystem-/Config-Reader ist der kleinste robuste Einstiegspunkt fuer Phase 1
  - die stabilsten Anker fuer die erste UI sind:
    - Label-Contracts unter `configs/label_contracts/`
    - kuratierte Kern-Doku
    - kuratierte `work/`-Roots und vorhandene Unterordner
    - config-referenzierte Pfade unter `configs/*.yaml`
  - fehlende lokale Artefaktpfade koennen sauber als `missing` angezeigt werden, ohne die UI zu brechen
- Real ausgefuehrt:
  - WebUI-Abhaengigkeiten im aktuellen Interpreter installiert
  - neue FastAPI-App importiert und via TestClient verifiziert
  - lokaler Uvicorn-Start gegen die echte Repo-App ausgefuehrt
  - HTTP-200-Probe auf `/` erfolgreich beantwortet
  - Ruff Format / Ruff Check / Pytest ausgefuehrt

## Tests
- `python -m pip install -r requirements/dev.txt`
  - Exit-Code: `0`
- `python -m ruff format .`
  - Exit-Code: `0`
  - Ergebnis: `6 files reformatted, 69 files left unchanged`
- `python -m ruff check .`
  - Exit-Code: `1`
  - Ergebnis: neue WebUI-Testdateien hatten unsortierte Imports
- `python -m ruff check tests/test_webui_app.py tests/test_webui_reader.py --fix`
  - Exit-Code: `0`
  - Ergebnis: `2` Importordnungen automatisch korrigiert
- `python -m ruff check .`
  - Exit-Code: `0`
  - Ergebnis: `All checks passed!`
- `python -m pytest`
  - Exit-Code: `1`
  - Ergebnis: ein WebUI-Test schlug fehl, weil der Contract-Status noch nicht auf der Contracts-Seite sichtbar war
- `python -m ruff format .`
  - Exit-Code: `0`
  - Ergebnis: `75 files left unchanged`
- `python -m ruff check .`
  - Exit-Code: `0`
  - Ergebnis: `All checks passed!`
- `python -m pytest`
  - Exit-Code: `0`
  - Ergebnis: `165 passed, 5 skipped in 3.63s`
- `PYTHONPATH=src python -m uvicorn owli_train.webui.app:app --host 127.0.0.1 --port 8000`
  - Exit-Code: `0` nach sauberem `CTRL+C`
  - Ergebnis: Uvicorn startete lokal auf `http://127.0.0.1:8000`
- 
  ```bash
  python - <<'PY'
  from urllib.request import urlopen
  with urlopen('http://127.0.0.1:8000/') as response:
      body = response.read(200).decode('utf-8', errors='replace')
      print(response.status)
      print('Owli Control UI' in body)
  PY
  ```
  - Exit-Code: `0`
  - Ergebnis: HTTP-Status `200`, erwarteter Seitentitel im Response-Body gefunden

## Relevante Run-Kommandos
- WSL2-Start der WebUI:
```bash
source .venv/bin/activate
PYTHONPATH=src python -m uvicorn owli_train.webui.app:app --host 127.0.0.1 --port 8000 --reload
```
- PowerShell-Start der WebUI:
```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m uvicorn owli_train.webui.app:app --host 127.0.0.1 --port 8000 --reload
```

## Offene Risiken
- Die WebUI ist bewusst nur read-only; es gibt noch keine echte Job-Status- oder Job-Steuerungsintegration ueber laufende Prozesse hinweg.
- Die Artefaktsicht basiert in Phase 1 auf kuratierten Repo- und `work/`-Pfaden; exotische oder neue Pfadkonventionen erscheinen erst, wenn sie explizit in den Reader aufgenommen werden.
- Es gibt absichtlich noch keinen Dateiinhalt-Viewer fuer Reports oder Artefakte im Browser, sondern nur eine uebersichtliche Pfad- und Statussicht.

## Naechster sinnvoller Schritt
- Ergaenze als naechsten kleinen Schritt eine read-only Run-Detail-Seite, die erkannte Run-Artefakte, Eval-Reports und Exportdateien pro `work/runs/<run_id>` gezielt zusammenfasst.
