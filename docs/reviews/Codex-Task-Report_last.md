# Codex Task Report

## Ziel
- Den ersten echten `EfficientDet-Lite2`-Trainingslauf auf dem vorbereiteten BA-v2-MVP-Kandidaten real ausfuehren.
- Die exportierten BA-v2-MVP-Artefakte mit `inspect tflite`, `eval efficientdet-tflite` und `golden detect` real verifizieren.
- Die Resultate als erste ehrliche BA-v2-MVP-Baseline im Repo dokumentieren und gegen die historischen Baselines einordnen.

## Was wurde geändert?
- Neue Ergebnisdoku fuer die erste echte BA-v2-MVP-Baseline ergaenzt:
  - `docs/BA_v2_MVP_Baseline.md`
- BA-v2-MVP-Plan und Candidate-Doku auf den jetzt real abgeschlossenen Lauf nachgezogen:
  - `docs/MVP_Training_Plan.md`
  - `docs/BA_v2_MVP_Train_Candidate.md`
- Runbook um die verifizierten BA-v2-MVP-Train/Eval/Golden-Kommandos erweitert:
  - `docs/runbook.md`
- Pflicht-Report fuer diesen Task aktualisiert:
  - `docs/reviews/Codex-Task-Report_last.md`

## Was wurde wirklich verifiziert?
- Statisch geprueft:
  - `README.md`
  - `docs/BA_v2_Hazard_Labelset.md`
  - `docs/BA_v2_MVP_Baseline.md`
  - `docs/BA_v2_MVP_Train_Candidate.md`
  - `docs/MVP_Training_Plan.md`
  - `docs/runbook.md`
  - `docs/android-export-contract.md`
  - `configs/label_contracts/ba_v2_hazard.yaml`
  - `configs/efficientdet_lite2_ba_v2_mvp.yaml`
  - BA-v2-relevante Merge-/Materialize-/CSV-Configs unter `configs/*`
  - Trainings-/Eval-/Golden-/TFLite-Pfade unter:
    - `src/owli_train/training/*`
    - `src/owli_train/eval/*`
    - `src/owli_train/golden/*`
    - `src/owli_train/tflite_detect.py`
- Inhaltlich verifiziert:
  - der reale BA-v2-MVP-Datensatz ist `work/datasets/ba_v2_mvp_candidate`
  - die reale Trainingsconfig ist `configs/efficientdet_lite2_ba_v2_mvp.yaml`
  - die BA-v2-MVP-Klassenreihenfolge blieb durch Training, Export und Eval unveraendert:
    - `obstacle_ground`
    - `obstacle_barrier`
    - `obstacle_hole_dropoff`
    - `obstacle_pole`
    - `person`
    - `bicycle`
    - `motorcycle`
    - `car`
    - `bus`
    - `truck`
  - `mapping_files.json` meldet keine fehlenden BA-v2-Klassen im `TRAIN`-Split
  - TFLite-Eval konnte die Kategorien direkt per `labels.txt` gegen den BA-v2-TEST-Split ausrichten
- Real ausgefuehrt:
  - echter BA-v2-MVP-Lite2-Trainingslauf mit GPU und ohne `--max-steps`
  - echtes Lite2-Export-Artefakt unter `work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/artifacts/model.tflite`
  - `inspect tflite` auf dem exportierten BA-v2-Modell
  - `eval efficientdet-tflite` auf dem kompletten BA-v2-`TEST`-Split mit `381` Bildern
  - `golden detect` auf einem BA-v2-`TEST`-Bild mit drei Hazard-Core-Klassen plus Rehearsal-Mix
  - erzeugte Artefakte:
    - Run dir: `work/runs/20260309-111756-ba-v2-mvp-baseline-20260309`
    - TFLite: `work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/artifacts/model.tflite`
    - Labels: `work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/artifacts/labels.txt`
    - Class names: `work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/artifacts/class_names.json`
    - Eval JSON: `work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/reports/eval_efficientdet_tflite_ba_v2_test.json`
    - Eval Markdown: `work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/reports/eval_efficientdet_tflite_ba_v2_test.md`
    - Golden JSON: `work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/reports/golden_ba_v2_test_mix.json`

## Tests
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet --config configs/efficientdet_lite2_ba_v2_mvp.yaml --run-name ba-v2-mvp-baseline-20260309 --require-gpu`
  - Exit-Code: `0`
  - Ergebnis: echter BA-v2-MVP-Lauf mit `20` Epochen, Export von `model.tflite`, `labels.txt`, `class_names.json`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite --model work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/artifacts/model.tflite`
  - Exit-Code: `0`
  - Ergebnis: `builtin_ops_only=true`, Input `448x448x3 uint8`, erwartete EfficientDet-Lite2-Operatoren
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite --coco work/splits/ba_v2_hazard_slice02_mapillary_od_ground/instances_test.json --images-dir work/datasets/ba_v2_mvp_candidate/images --model work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/artifacts/model.tflite --score-threshold 0.1 --noise-thresholds 0.05,0.1,0.3 --num-threads 8 --out work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/reports/eval_efficientdet_tflite_ba_v2_test.json`
  - Exit-Code: `0`
  - Ergebnis: kompletter BA-v2-`TEST`-Split (`381` Bilder) ausgewertet, AP `0.1184`, AP50 `0.2149`, AP75 `0.1120`, AR100 `0.2005`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect --model work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/artifacts/model.tflite --image work/datasets/ba_v2_mvp_candidate/images/mapillary_vistas/training/ppvi1a8kNPmFjkS6Lhbnsg.jpg --out work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/reports/golden_ba_v2_test_mix.json --score-threshold 0.1 --max-results 20 --num-threads 8`
  - Exit-Code: `0`
  - Ergebnis: `20` Detections geschrieben; Klassenmix `person=11`, `car=8`, `truck=1`
- Nicht ausgefuehrt:
  - `ruff format`, `ruff check`, `pytest`
  - Grund: nach dem Lauf wurden nur Doku-/Report-Dateien aktualisiert; die reale Verifikation fuer diesen Task lag bewusst auf dem echten Train/Eval/Golden-Pfad

## Relevante Run-Kommandos
- Train BA-v2 MVP baseline:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet \
  --config configs/efficientdet_lite2_ba_v2_mvp.yaml \
  --run-name ba-v2-mvp-baseline-20260309 \
  --require-gpu
```
- Inspect exported BA-v2 Lite2 model:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite \
  --model work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/artifacts/model.tflite
```
- Eval exported BA-v2 Lite2 model on the held-out BA-v2 `TEST` split:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_v2_hazard_slice02_mapillary_od_ground/instances_test.json \
  --images-dir work/datasets/ba_v2_mvp_candidate/images \
  --model work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 8 \
  --out work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/reports/eval_efficientdet_tflite_ba_v2_test.json
```
- Generate BA-v2 golden detect sample:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect \
  --model work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/artifacts/model.tflite \
  --image work/datasets/ba_v2_mvp_candidate/images/mapillary_vistas/training/ppvi1a8kNPmFjkS6Lhbnsg.jpg \
  --out work/runs/20260309-111756-ba-v2-mvp-baseline-20260309/reports/golden_ba_v2_test_mix.json \
  --score-threshold 0.1 \
  --max-results 20 \
  --num-threads 8
```

## Offene Risiken
- `obstacle_ground` bleibt datenmaessig ein schmaler Legacy-Bootstrap aus `Obstacle4` und ist quantitativ noch nicht brauchbar.
- `obstacle_barrier` und `obstacle_hole_dropoff` zeigen zwar Lernsignal, sind aber weiter klar zu schwach und zu FP-lastig fuer einen produktnahen Hazard-Readout.
- `obstacle_pole` bleibt trotz vieler Annotationen schwierig; die aktuelle BA-v2-Baseline deutet eher auf Ambiguitaet / Hintergrundclutter als auf ein reines Mengenproblem.
- Die qualitative Golden-Probe auf einem Bild mit drei Hazard-Core-Klassen wird im Top-20-Output von `person` / `car` dominiert und zeigt keine Hazard-Core-Dimension im geschriebenen Sample.
- Der historische Stage-3-BA-v1-Lauf bleibt der staerkere technische Vergleichsanker; die neue BA-v2-Baseline ist in erster Linie Produktlogik-Evidenz plus erster echter End-to-End-Nachweis.

## Nächster sinnvoller Schritt
- Verbessere innerhalb des bestehenden BA-v2-MVP-Contracts gezielt die Hazard-Core-Datenqualitaet und FP-Kontrolle, statt den Scope erneut zu erweitern.
