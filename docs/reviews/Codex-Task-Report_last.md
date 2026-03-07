# Codex Task Report

## Ziel
- Den korrigierten Obstacle4-Produktionspfad auf repo HEAD einmal real bis zu einem voll exportierten EfficientDet-Lite2-TFLite-Ergebnis durchlaufen lassen und die Produktdoku auf Basis der echten Resultate aktualisieren.

## Was wurde geändert?
- Es wurden keine Produktions-Codepfade geändert.
- `docs/Obstacle4_E2E_Results.md` wurde auf den echten Obstacle4-Vollrun `20260307-222245-obstacle4-e2e-20260307` aktualisiert.
- `docs/runbook.md` wurde im Obstacle4-Abschnitt knapp auf den tatsächlich verifizierten Full-Eval-/Golden-Ablauf mit Report-Ausgabe und `--num-threads 8` geschärft.
- `docs/reviews/Codex-Task-Report_last.md` wurde auf diesen Taskstand fortgeschrieben.
- Geänderte Dateien:
- `docs/Obstacle4_E2E_Results.md`
- `docs/runbook.md`
- `docs/reviews/Codex-Task-Report_last.md`

## Was wurde wirklich verifiziert?
- Tatsächlich ausgeführte Kommandos:
- `python -m owli_train dataset import yolo --yolo-dir data/raw/obstacle4/extracted --out work/datasets/obstacle4/instances_raw.json`
- Exit-Code: `0`
- Ergebnis: `1250` Bilder, `1627` Annotationen, `4` Quellkategorien.
- `python -m owli_train dataset normalize --coco work/datasets/obstacle4/instances_raw.json --images-dir data/raw/obstacle4/extracted --label-map configs/label_maps/obstacle4_to_ba.yaml --out work/datasets/obstacle4/instances_gt.json`
- Exit-Code: `0`
- Ergebnis: normalisierte BA-GT-Datei geschrieben.
- `python -m owli_train dataset validate --coco work/datasets/obstacle4/instances_gt.json --images-dir data/raw/obstacle4/extracted`
- Exit-Code: `0`
- Ergebnis: GT-COCO gegen Images validiert.
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train dataset pseudo-label coco --images-dir data/raw/obstacle4/extracted --out work/datasets/obstacle4/pseudo_coco_critical.json --classes person,bicycle,motorcycle,car,bus,truck --score-threshold 0.45 --batch-size 1`
- Exit-Code: `0`
- Ergebnis: `1250` Bilder verarbeitet, `288` Detections behalten, darunter `bus=4`.
- `python -m owli_train dataset merge coco --manifest configs/merge_obstacle4_gt_pseudo.yaml --out work/datasets/obstacle4/instances_combined.json`
- Exit-Code: `0`
- Ergebnis: `1250` Bilder, `1912` Annotationen, `10` Kategorien.
- `python -m owli_train dataset split --coco work/datasets/obstacle4/instances_combined.json --out-dir work/splits/obstacle4 --seed 1337 --ensure-train-class-coverage`
- Exit-Code: `0`
- Ergebnis: deterministischer Split mit TRAIN-Class-Coverage erzeugt.
- `python -m owli_train dataset export modelmaker-csv --coco work/datasets/obstacle4/instances_combined.json --images-dir data/raw/obstacle4/extracted --splits-json work/splits/obstacle4/splits.json --out work/datasets/obstacle4/modelmaker.csv`
- Exit-Code: `0`
- Ergebnis: ModelMaker-CSV exportiert; alle `10` erwarteten Klassen in TRAIN vorhanden.
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet --config configs/efficientdet_lite2_obstacle4.yaml --run-name obstacle4-e2e-20260307 --require-gpu`
- Exit-Code: `0`
- Ergebnis: voller EfficientDet-Lite2-Trainingslauf abgeschlossen; Export nach `work/runs/20260307-222245-obstacle4-e2e-20260307/artifacts/model.tflite`.
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite --model work/runs/20260307-222245-obstacle4-e2e-20260307/artifacts/model.tflite`
- Exit-Code: `0`
- Ergebnis: builtin-ops-only TFLite, Modellgroesse `7.1 MB`, finaler 10-Klassen-Contract intakt.
- Erster Voll-Eval-Versuch ohne `--num-threads` wurde real gestartet, nach mehreren Minuten Single-Core-Lauf aber bewusst abgebrochen und nicht fuer Resultate verwendet.
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite --coco work/datasets/obstacle4/instances_combined.json --images-dir data/raw/obstacle4/extracted --model work/runs/20260307-222245-obstacle4-e2e-20260307/artifacts/model.tflite --score-threshold 0.1 --noise-thresholds 0.05,0.1,0.3 --num-threads 8 --out work/runs/20260307-222245-obstacle4-e2e-20260307/reports/eval_efficientdet_tflite.json`
- Exit-Code: `0`
- Ergebnis: voller TFLite-Eval ueber `1250` Bilder; AP `0.0952`, AP50 `0.1897`, AP75 `0.0886`, `FP/100=984.8` bei Schwelle `0.1`.
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect --model work/runs/20260307-222245-obstacle4-e2e-20260307/artifacts/model.tflite --image data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg --out work/runs/20260307-222245-obstacle4-e2e-20260307/reports/golden_obstacle4.json --score-threshold 0.1 --max-results 20 --num-threads 8`
- Exit-Code: `0`
- Ergebnis: `7` Detections auf dem Referenzbild geschrieben.
- `python -m ruff format .`
- Exit-Code: `0`
- Ergebnis: `56 files left unchanged`.
- `python -m ruff check .`
- Exit-Code: `0`
- Ergebnis: `All checks passed!`.
- `python -m pytest`
- Exit-Code: `0`
- Ergebnis: `116 passed, 5 skipped in 1.82s`.
- Rein statisch geprüft:
- `labels.txt`, `class_names.json` und `mapping_files.json` wurden lokal gelesen und auf denselben 10-Klassen-Contract geprüft.

## Tests
- `python -m ruff format .`
- Exit-Code: `0`
- Resultat: `56 files left unchanged`.
- `python -m ruff check .`
- Exit-Code: `0`
- Resultat: `All checks passed!`.
- `python -m pytest`
- Exit-Code: `0`
- Resultat: `116 passed, 5 skipped in 1.82s`.
- Zusaetzlich wurden die realen Obstacle4-End-to-End-Kommandos mit Exit-Code `0` als primäre Verifikation des Produktionspfads ausgeführt.

## Relevante Run-Kommandos
- `python -m owli_train dataset import yolo --yolo-dir data/raw/obstacle4/extracted --out work/datasets/obstacle4/instances_raw.json`
- `python -m owli_train dataset normalize --coco work/datasets/obstacle4/instances_raw.json --images-dir data/raw/obstacle4/extracted --label-map configs/label_maps/obstacle4_to_ba.yaml --out work/datasets/obstacle4/instances_gt.json`
- `python -m owli_train dataset validate --coco work/datasets/obstacle4/instances_gt.json --images-dir data/raw/obstacle4/extracted`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train dataset pseudo-label coco --images-dir data/raw/obstacle4/extracted --out work/datasets/obstacle4/pseudo_coco_critical.json --classes person,bicycle,motorcycle,car,bus,truck --score-threshold 0.45 --batch-size 1`
- `python -m owli_train dataset merge coco --manifest configs/merge_obstacle4_gt_pseudo.yaml --out work/datasets/obstacle4/instances_combined.json`
- `python -m owli_train dataset split --coco work/datasets/obstacle4/instances_combined.json --out-dir work/splits/obstacle4 --seed 1337 --ensure-train-class-coverage`
- `python -m owli_train dataset export modelmaker-csv --coco work/datasets/obstacle4/instances_combined.json --images-dir data/raw/obstacle4/extracted --splits-json work/splits/obstacle4/splits.json --out work/datasets/obstacle4/modelmaker.csv`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet --config configs/efficientdet_lite2_obstacle4.yaml --run-name obstacle4-e2e-20260307 --require-gpu`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite --model work/runs/20260307-222245-obstacle4-e2e-20260307/artifacts/model.tflite`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite --coco work/datasets/obstacle4/instances_combined.json --images-dir data/raw/obstacle4/extracted --model work/runs/20260307-222245-obstacle4-e2e-20260307/artifacts/model.tflite --score-threshold 0.1 --noise-thresholds 0.05,0.1,0.3 --num-threads 8 --out work/runs/20260307-222245-obstacle4-e2e-20260307/reports/eval_efficientdet_tflite.json`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect --model work/runs/20260307-222245-obstacle4-e2e-20260307/artifacts/model.tflite --image data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg --out work/runs/20260307-222245-obstacle4-e2e-20260307/reports/golden_obstacle4.json --score-threshold 0.1 --max-results 20 --num-threads 8`

## Offene Risiken
- Der 10-Klassen-Contract bleibt zwar bis in das finale Lite2-Modell erhalten, aber `person`, `bicycle`, `motorcycle`, `bus` und `truck` werden im realen Voll-Eval praktisch nicht gelernt.
- Die False-Positive-Last ist fuer BA-Anwendungsfaelle noch hoch, selbst wenn einige BA-Klassen bereits brauchbare Recall-Werte zeigen.
- Der Golden-Output ist nur ein Einzelbild und kein Ersatz fuer eine systematische On-Device- oder Szenario-Pruefung.
- Der in diesem Task verifizierte Lauf ist deshalb ein technischer Referenzbaseline, kein Nachweis von Produktreife.

## Nächster sinnvoller Schritt
- Einen gezielten BA-Datensatzschritt fuer die schwachen COCO-kritischen Klassen aufsetzen, damit `person`, `bicycle`, `motorcycle`, `bus` und `truck` im naechsten Obstacle4-Lite2-Run nicht nur im Label-Contract existieren, sondern auch real gelernt werden.
