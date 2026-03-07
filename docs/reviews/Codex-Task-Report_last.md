# Codex Task Report

## Ziel
- Das aktuell sichtbare Obstacle4-Datenproblem so beheben, dass alle erwarteten Klassen des produktiven EfficientDet-Lite2-/ModelMaker-Contracts wieder im TRAIN-Split vertreten sind und der Trainings-Preflight ohne Opt-out durchlaeuft.

## Was wurde geändert?
- In `src/owli_train/data/split.py` wurde mit `ensure_train_split_class_coverage(...)` ein kleiner, deterministischer Split-Repair ergaenzt, der vorhandene, aber im TRAIN fehlende Klassen aus `VAL/TEST` nach `TRAIN` ziehen kann.
- `split_coco_image_ids(...)` unterstuetzt jetzt optional `ensure_train_class_coverage=True`.
- Die CLI `dataset split` in `src/owli_train/cli.py` hat dafuer das Flag `--ensure-train-class-coverage` erhalten.
- In `configs/merge_obstacle4_gt_pseudo.yaml` wurde der produktive Pseudo-Score-Threshold von `0.6` auf `0.45` gesenkt, weil `0.6` auf dem aktuellen Obstacle4-Bestand real `0` `bus`-Detektionen erzeugt.
- `docs/runbook.md` und `docs/Obstacle4_E2E_Results.md` wurden auf den verifizierten Obstacle4-Ablauf aktualisiert: Pseudo-Labels mit `0.45`, Merge vor Split, Split auf `instances_combined.json` mit `--ensure-train-class-coverage`.
- In `tests/test_dataset_split.py` wurden gezielte Tests fuer den Coverage-Repair und das neue CLI-Flag ergaenzt.
- Geaenderte Dateien:
- `src/owli_train/data/split.py`
- `src/owli_train/cli.py`
- `tests/test_dataset_split.py`
- `configs/merge_obstacle4_gt_pseudo.yaml`
- `docs/runbook.md`
- `docs/Obstacle4_E2E_Results.md`

## Was wurde wirklich verifiziert?
- Tatsächlich ausgefuehrte Kommandos:
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train dataset pseudo-label coco --images-dir data/raw/obstacle4/extracted --out work/datasets/obstacle4/pseudo_coco_critical_t03.json --classes person,bicycle,motorcycle,car,bus,truck --score-threshold 0.3 --batch-size 1`
- `python -m pytest tests/test_dataset_split.py tests/test_train_efficientdet_pipeline.py tests/test_train_efficientdet_cli.py tests/test_train_efficientdet_config.py tests/test_cli_help.py`
- `cp work/datasets/obstacle4/pseudo_coco_critical_t03.json work/datasets/obstacle4/pseudo_coco_critical.json`
- `python -m owli_train dataset merge coco --manifest configs/merge_obstacle4_gt_pseudo.yaml --out work/datasets/obstacle4/instances_combined.json`
- `python -m owli_train dataset split --coco work/datasets/obstacle4/instances_combined.json --out-dir work/splits/obstacle4 --seed 1337 --ensure-train-class-coverage`
- `python -m owli_train dataset export modelmaker-csv --coco work/datasets/obstacle4/instances_combined.json --images-dir data/raw/obstacle4/extracted --splits-json work/splits/obstacle4/splits.json --out work/datasets/obstacle4/modelmaker.csv`
- `env -u MODELMAKER_PYTHON_EXE PYTHONPATH=src python -m owli_train train efficientdet --config configs/efficientdet_lite2_obstacle4.yaml`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet --config configs/efficientdet_lite2_obstacle4.yaml --max-steps 1 --run-name obstacle4-class-coverage-smoke --require-gpu`
- `python -m ruff format .`
- `python -m ruff check .`
- `python -m pytest`
- Rein statisch geprueft:
- Die Label-Contract-Herkunft in `configs/efficientdet_lite2_obstacle4.yaml` gegen `work/datasets/obstacle4/modelmaker.class_names.json`.
- Die aktualisierten Doku-Kommandos gegen die aktuelle CLI-Syntax.

## Tests
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train dataset pseudo-label coco --images-dir data/raw/obstacle4/extracted --out work/datasets/obstacle4/pseudo_coco_critical_t03.json --classes person,bicycle,motorcycle,car,bus,truck --score-threshold 0.3 --batch-size 1`
- Exit-Code: `0`
- Resultat: echter Teacher-Run auf GPU; `1250` Bilder verarbeitet, `530` Detections behalten. Daraus verifiziert: bei Threshold `0.45` bleiben `4` `bus`-Detektionen erhalten.
- `python -m pytest tests/test_dataset_split.py tests/test_train_efficientdet_pipeline.py tests/test_train_efficientdet_cli.py tests/test_train_efficientdet_config.py tests/test_cli_help.py`
- Exit-Code: `0`
- Resultat: `34 passed in 2.15s`.
- `python -m owli_train dataset merge coco --manifest configs/merge_obstacle4_gt_pseudo.yaml --out work/datasets/obstacle4/instances_combined.json`
- Exit-Code: `0`
- Resultat: `annotations=1912`, `categories=10`.
- `python -m owli_train dataset split --coco work/datasets/obstacle4/instances_combined.json --out-dir work/splits/obstacle4 --seed 1337 --ensure-train-class-coverage`
- Exit-Code: `0`
- Resultat: `splits.json` neu geschrieben.
- `python -m owli_train dataset export modelmaker-csv --coco work/datasets/obstacle4/instances_combined.json --images-dir data/raw/obstacle4/extracted --splits-json work/splits/obstacle4/splits.json --out work/datasets/obstacle4/modelmaker.csv`
- Exit-Code: `0`
- Resultat: `rows=1912`; anschliessende CSV-Inspektion zeigte im `TRAIN`-Split alle 10 erwarteten Klassen, darunter `bus: 4`.
- `env -u MODELMAKER_PYTHON_EXE PYTHONPATH=src python -m owli_train train efficientdet --config configs/efficientdet_lite2_obstacle4.yaml`
- Exit-Code: `1`
- Resultat: kein Missing-Class-Gate mehr; der Lauf faellt erst am erwarteten fehlenden ModelMaker-Dependency-Load in der Haupt-venv.
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet --config configs/efficientdet_lite2_obstacle4.yaml --max-steps 1 --run-name obstacle4-class-coverage-smoke --require-gpu`
- Exit-Code: `0`
- Resultat: echter py39-ModelMaker-Smoke-Run erfolgreich, Run `work/runs/20260307-215618-obstacle4-class-coverage-smoke` erzeugt. Hinweis: der kleine `--max-steps 1`-Subset meldete erwartbar `missing_classes_from_training: bicycle, bus`; das betrifft nur den Smoke-Subset, nicht den vollen Obstacle4-CSV-Contract.
- `python -m ruff format .`
- Exit-Code: `0`
- Resultat: `1 file reformatted`.
- `python -m ruff check .`
- Exit-Code: `0`
- Resultat: `All checks passed!`.
- `python -m pytest`
- Exit-Code: `0`
- Resultat: `114 passed, 5 skipped in 1.76s`.

## Relevante Run-Kommandos
- Produktiver Obstacle4-Pfad unter WSL2:
- `python -m owli_train dataset import yolo --yolo-dir data/raw/obstacle4/extracted --out work/datasets/obstacle4/instances_raw.json`
- `python -m owli_train dataset normalize --coco work/datasets/obstacle4/instances_raw.json --images-dir data/raw/obstacle4/extracted --label-map configs/label_maps/obstacle4_to_ba.yaml --out work/datasets/obstacle4/instances_gt.json`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train dataset pseudo-label coco --images-dir data/raw/obstacle4/extracted --out work/datasets/obstacle4/pseudo_coco_critical.json --classes person,bicycle,motorcycle,car,bus,truck --score-threshold 0.45 --batch-size 1`
- `python -m owli_train dataset merge coco --manifest configs/merge_obstacle4_gt_pseudo.yaml --out work/datasets/obstacle4/instances_combined.json`
- `python -m owli_train dataset split --coco work/datasets/obstacle4/instances_combined.json --out-dir work/splits/obstacle4 --seed 1337 --ensure-train-class-coverage`
- `python -m owli_train dataset export modelmaker-csv --coco work/datasets/obstacle4/instances_combined.json --images-dir data/raw/obstacle4/extracted --splits-json work/splits/obstacle4/splits.json --out work/datasets/obstacle4/modelmaker.csv`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet --config configs/efficientdet_lite2_obstacle4.yaml --require-gpu`

## Offene Risiken
- Die produktive Coverage ist jetzt fuer den aktuellen Obstacle4-Stand verifiziert, aber die Pseudo-Labels fuer `bus` bleiben schwach besetzt; bei kuenftigen Teacher-/Threshold-Aenderungen kann die Klasse erneut wegbrechen.
- Der echte Voll-Trainingslauf ohne `--max-steps` wurde in diesem Task nicht erneut komplett ausgefuehrt.
- Der neue Split-Repair schuetzt nur Klassen, die im kombinierten COCO tatsaechlich mindestens eine Annotation haben; bei real `0` Vorkommen bleibt der Trainings-Gate weiterhin korrekt blockierend.

## Nächster sinnvoller Schritt
- Einen vollstaendigen Obstacle4-Lite2-Lauf auf dem jetzt korrigierten Pfad ohne `--max-steps` ausfuehren und die resultierenden Eval-/Golden-Kennzahlen in den Produktdokumenten aktualisieren.
