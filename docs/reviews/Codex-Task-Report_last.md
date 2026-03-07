# Codex Task Report

## Ziel
- Den produktionsnahen Obstacle4/EfficientDet-Lite2-Pfad auf dem Repo-Stand vom 2026-03-07 verifizieren und mit kleinen, reviewbaren Änderungen robuster machen.
- Fokus: Reproduzierbarkeit unter WSL2, Klassenkonsistenz, korrekte Label-Auflösung über Import, Merge, ModelMaker-CSV, Training, TFLite-Eval und Golden Detect.

## Was wurde geändert?
- Die produktionsrelevante CLI wurde in `src/owli_train/cli.py` auf interpreter-konsistente Required-Optionen umgestellt, damit Haupt-Interpreter und ModelMaker-py39 dieselben Aufrufe akzeptieren.
- Die Delegation über `MODELMAKER_PYTHON_EXE` und `TEACHER_PYTHON_EXE` wurde für `eval efficientdet-tflite`, `golden detect` und Teacher-Pseudo-Labeling auf dieselbe Flag-Syntax gebracht.
- Der Teacher-Pfad in `src/owli_train/pseudo_label/teacher_tfhub.py` erkennt jetzt feste Batch-Dimensionen aus der Signatur und reduziert die effektive Batch-Größe automatisch; bei verbleibendem Multi-Batch-Fehler kommt eine klare Diagnose.
- Die Trainings-CLI gibt fehlende Klassen aus `mapping_files.json` direkt als Warnung aus, wenn Klassen im TRAIN-Split nicht vorkommen.
- Die Obstacle4-Dokumentation wurde auf den verifizierten Repo-Stand gebracht, inklusive fehlendem `dataset split`-Schritt und aktueller CLI-Beispiele.
- Geänderte Dateien:
- `src/owli_train/cli.py`
- `src/owli_train/pseudo_label/teacher_tfhub.py`
- `docs/runbook.md`
- `docs/Obstacle4_E2E_Results.md`
- `tests/test_pseudo_label_cli.py`
- `tests/test_eval_efficientdet_tflite_cli.py`
- `tests/test_golden_cli.py`
- `tests/test_teacher_tfhub.py`

## Was wurde wirklich verifiziert?
- Tatsächlich ausgeführte Kommandos:
- `python -m owli_train dataset validate --coco work/datasets/obstacle4/instances_gt.json --images-dir data/raw/obstacle4/extracted`
- `python -m owli_train dataset validate --coco work/datasets/obstacle4/instances_combined.json --images-dir data/raw/obstacle4/extracted`
- `python -m owli_train dataset import yolo --yolo-dir data/raw/obstacle4/extracted --out work/verify_obstacle4/instances_raw.json`
- `python -m owli_train dataset normalize --coco work/verify_obstacle4/instances_raw.json --images-dir data/raw/obstacle4/extracted --label-map configs/label_maps/obstacle4_to_ba.yaml --out work/verify_obstacle4/instances_gt.json`
- `python -m owli_train dataset split --coco work/verify_obstacle4/instances_gt.json --out-dir work/verify_obstacle4/splits --seed 1337`
- `python -m owli_train dataset merge coco --manifest configs/merge_obstacle4_gt_pseudo.yaml --out work/verify_obstacle4/instances_combined.json --report-out work/verify_obstacle4/instances_combined.report.json`
- `python -m owli_train dataset export modelmaker-csv --coco work/verify_obstacle4/instances_combined.json --images-dir data/raw/obstacle4/extracted --splits-json work/verify_obstacle4/splits/splits.json --out work/verify_obstacle4/modelmaker.csv`
- `python -m owli_train dataset materialize-images --coco work/verify_obstacle4/instances_combined.json --merge-manifest configs/merge_obstacle4_gt_pseudo.yaml --out-images-dir work/verify_obstacle4/materialized/images --out-coco work/verify_obstacle4/instances_materialized.json --mode symlink`
- `.venv-modelmaker-py39/bin/python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite --model work/runs/20260216-192857/artifacts/model.tflite`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect --model work/runs/20260216-192857/artifacts/model.tflite --image data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg --out work/verify_obstacle4/golden_obstacle4.json --score-threshold 0.1 --max-results 20`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite --coco work/datasets/obstacle4/instances_combined.json --images-dir data/raw/obstacle4/extracted --model work/runs/20260216-192857/artifacts/model.tflite --limit-images 5 --score-threshold 0.1 --noise-thresholds 0.05,0.1,0.3 --out work/verify_obstacle4/eval_efficientdet_tflite_5_noise.json`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet --config configs/efficientdet_lite2_obstacle4.yaml --max-steps 1 --subset-seed 1337 --run-name verify-obstacle4-smoke --require-gpu`
- Rein statisch geprüft:
- CLI-Hilfen und Signaturen der Commands in Haupt-Interpreter und ModelMaker-py39.
- Konsistenz zwischen `configs/efficientdet_lite2_obstacle4.yaml`, `configs/merge_obstacle4_gt_pseudo.yaml`, `configs/label_maps/obstacle4_to_ba.yaml`, den bestehenden Run-Artefakten und der Doku.
- Label-Reihenfolge und Mapping-Artefakte in bestehenden Runs per Datei-Inspektion, nicht durch kompletten neuen Voll-Trainingslauf.

## Tests
- `python -m ruff format src/owli_train/cli.py tests/test_pseudo_label_cli.py tests/test_eval_efficientdet_tflite_cli.py tests/test_golden_cli.py tests/test_teacher_tfhub.py`
- Exit-Code: `0`
- Resultat: Formatierung erfolgreich.
- `python -m ruff check .`
- Exit-Code: `0`
- Resultat: Alle Checks bestanden.
- `python -m pytest`
- Exit-Code: `0`
- Resultat: `107 passed, 5 skipped`.
- Die Tests wurden nach den finalen Codeänderungen erneut ausgeführt.

## Relevante Run-Kommandos
- `python -m owli_train dataset import yolo --yolo-dir data/raw/obstacle4/extracted --out work/datasets/obstacle4/instances_raw.json`
- `python -m owli_train dataset normalize --coco work/datasets/obstacle4/instances_raw.json --images-dir data/raw/obstacle4/extracted --label-map configs/label_maps/obstacle4_to_ba.yaml --out work/datasets/obstacle4/instances_gt.json`
- `python -m owli_train dataset split --coco work/datasets/obstacle4/instances_gt.json --out-dir work/splits/obstacle4 --seed 1337`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train dataset pseudo-label coco --images-dir data/raw/obstacle4/extracted --out work/datasets/obstacle4/pseudo_coco_critical.json --classes person,bicycle,motorcycle,car,bus,truck --score-threshold 0.6 --batch-size 1`
- `python -m owli_train dataset merge coco --manifest configs/merge_obstacle4_gt_pseudo.yaml --out work/datasets/obstacle4/instances_combined.json`
- `python -m owli_train dataset export modelmaker-csv --coco work/datasets/obstacle4/instances_combined.json --images-dir data/raw/obstacle4/extracted --splits-json work/splits/obstacle4/splits.json --out work/datasets/obstacle4/modelmaker.csv`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet --config configs/efficientdet_lite2_obstacle4.yaml --require-gpu`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite --model work/runs/<run_id>/artifacts/model.tflite`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite --coco work/datasets/obstacle4/instances_combined.json --images-dir data/raw/obstacle4/extracted --model work/runs/<run_id>/artifacts/model.tflite --limit-images 50 --score-threshold 0.1 --noise-thresholds 0.05,0.1,0.3`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect --model work/runs/<run_id>/artifacts/model.tflite --image data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg --out work/runs/<run_id>/reports/golden_obstacle4.json --score-threshold 0.1 --max-results 20`

## Offene Risiken
- Es wurde kein kompletter neuer Obstacle4-End-to-End-Lauf mit vollem Pseudo-Labeling und vollem 20-Epochen-Training durchgeführt; verifiziert ist ein realistischer Smoke-Pfad plus bestehende Produktartefakte.
- Klassen, die im TRAIN-Split nicht vorkommen, fehlen weiterhin in den finalen TFLite-Labels. Das wird jetzt sichtbar gemacht, aber nicht verhindert.
- Der neue Teacher-Batch-Fallback ist durch Unit-Tests und Signaturerkennung abgesichert, aber nicht mit einem realen batch-1-only SavedModel gegen den kompletten Pseudo-Label-Run nachgestellt worden.
- Android-Laufzeit selbst wurde nicht getestet; verifiziert wurden TFLite-Inspect, Golden Detect und TFLite-Eval unter WSL2.

## Nächster sinnvoller Schritt
- Einen kleinen Trainings-Gate ergänzen, der den Obstacle4-Pfad fail-fast oder mindestens CI-sichtbar macht, wenn produktrelevante Klassen im TRAIN-Split fehlen und deshalb nicht in den finalen Lite2-Labels landen.
