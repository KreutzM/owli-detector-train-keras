# Codex Task Report

## Ziel
- Das erste BA-v1-Labelset im Repo explizit festlegen, die Rollen der Klassen dokumentieren, den naechsten Datensatzschritt priorisieren und den Contract mit minimalem Scope gegen Drift absichern.

## Was wurde geändert?
- Die neue BA-v1-Produktspezifikation wurde in `docs/BA_v1_Labelset.md` angelegt.
- Der BA-v1-Contract wurde minimal maschinenlesbar in `configs/label_contracts/ba_v1.yaml` eingefroren.
- `README.md`, `docs/runbook.md` und `docs/android-export-contract.md` verweisen jetzt knapp auf den BA-v1-Contract.
- In `tests/test_ba_v1_label_contract.py` wurde ein kleiner Drift-Test ergaenzt, der Reihenfolge, Rollenpartition und die Obstacle4->BA-Kernklassen-Zuordnung absichert.
- `docs/reviews/Codex-Task-Report_last.md` wurde auf diesen Taskstand fortgeschrieben.
- Geänderte Dateien:
- `configs/label_contracts/ba_v1.yaml`
- `docs/BA_v1_Labelset.md`
- `README.md`
- `docs/runbook.md`
- `docs/android-export-contract.md`
- `tests/test_ba_v1_label_contract.py`
- `docs/reviews/Codex-Task-Report_last.md`

## Was wurde wirklich verifiziert?
- Tatsächlich ausgeführte Kommandos:
- `sed -n '1,240p' README.md`
- `sed -n '1,260p' docs/Obstacle4_E2E_Results.md`
- `sed -n '1,260p' docs/runbook.md`
- `sed -n '1,260p' docs/android-export-contract.md`
- `sed -n '1,220p' configs/label_maps/obstacle4_to_ba.yaml`
- `sed -n '1,220p' configs/merge_obstacle4_gt_pseudo.yaml`
- `sed -n '1,220p' configs/efficientdet_lite2_obstacle4.yaml`
- `sed -n '1,220p' src/owli_train/data/modelmaker_csv.py`
- `sed -n '1,260p' src/owli_train/training/modelmaker_efficientdet.py`
- `sed -n '1,260p' src/owli_train/tflite_detect.py`
- `sed -n '1,260p' src/owli_train/eval/efficientdet_tflite.py`
- `sed -n '1,240p' src/owli_train/golden/detect.py`
- `rg -n ... src/owli_train tests`
- `python -m pytest tests/test_ba_v1_label_contract.py`
- Exit-Code: `0`
- Ergebnis: `3 passed in 0.10s`.
- `python -m ruff format .`
- Exit-Code: `0`
- Ergebnis: `57 files left unchanged`.
- `python -m ruff check .`
- Exit-Code: `0`
- Ergebnis: `All checks passed!`.
- `python -m pytest`
- Exit-Code: `0`
- Ergebnis: `119 passed, 5 skipped in 4.59s`.
- Ergebnis: der aktuelle Produktpfad nutzt faktisch einen 10-Klassen-Contract aus 4 BA-Kernklassen plus 6 COCO-kritischen Rehearsal-Klassen; die Reihenfolge wird ueber `class_names.json`, `labels.txt` und den TFLite-Resolver weitergetragen.
- Rein statisch geprüft:
- Der neue BA-v1-Contract und die Datensatzpriorisierung wurden gegen den aktuellen Repo-Stand und die letzte verifizierte Obstacle4-Baseline formuliert.
- Es wurde in diesem Task kein neuer Trainings- oder Evaluationslauf gestartet.

## Tests
- `python -m pytest tests/test_ba_v1_label_contract.py`
- Exit-Code: `0`
- Resultat: `3 passed in 0.10s`.
- `python -m ruff format .`
- Exit-Code: `0`
- Resultat: `57 files left unchanged`.
- `python -m ruff check .`
- Exit-Code: `0`
- Resultat: `All checks passed!`.
- `python -m pytest`
- Exit-Code: `0`
- Resultat: `119 passed, 5 skipped in 4.59s`.

## Relevante Run-Kommandos
- Kein neuer Laufpfad eingefuehrt.
- Relevante Dateien fuer den BA-v1-Contract:
- `configs/label_contracts/ba_v1.yaml`
- `docs/BA_v1_Labelset.md`
- Der bestehende Obstacle4-Referenzpfad bleibt in `docs/Obstacle4_E2E_Results.md` und `docs/runbook.md` dokumentiert.

## Offene Risiken
- Der BA-v1-Contract ist jetzt explizit, aber die Rehearsal-Klassen bleiben datenmaessig schwach; der Contract loest das Datenproblem nicht.
- Die Priorisierung `Obstacle-Dataset` vor `TACO` ist produktseitig sinnvoll, aber die genauen Quell-Taxonomien und Mappings sind im Repo noch nicht verifiziert.
- `TACO` ist in dieser Planung bewusst nur als Kandidat mit Annahmen beschrieben; daraus darf noch kein faktischer Mapping-Anspruch abgeleitet werden.

## Nächster sinnvoller Schritt
- Den naechsten PR auf die konkrete Integration des priorisierten Obstacle-Datasets zuschneiden und zuerst pruefen, wie sauber dessen Rohklassen in die vier BA-Kernklassen von BA-v1 gemappt werden koennen.
