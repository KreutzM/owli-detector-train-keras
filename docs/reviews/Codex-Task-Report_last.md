# Codex Task Report

## Ziel
- Den ersten echten `EfficientDet-Lite2`-Trainingslauf auf dem balancierten Multi-Source-Stage-3-Datensatz ausfuehren.
- Den Run bis `TFLite`, `inspect`, `eval` und `golden detect` real verifizieren.
- Die Resultate als neue Stage-3-Baseline dokumentieren, noch ohne `COCO replay`.

## Was wurde geändert?
- Neue dedizierte Stage-3-Trainingsconfig:
  - `configs/efficientdet_lite2_ba_mvp_stage3.yaml`
- Neue kanonische JSON-Label-Map fuer den ModelMaker-Pfad:
  - `configs/label_contracts/ba_v1.class_names.json`
- Kleiner produktionsrelevanter Fix im `OD`-Import:
  - `src/owli_train/data/obstacle_dataset.py`
  - Nicht-JPEG-Bildbytes mit `.jpg`-Dateinamen werden beim Export in echte JPEGs transkodiert.
- Neuer Test fuer diesen Fix:
  - `tests/test_dataset_import_obstacle_dataset.py`
- Neue Ergebnisdoku fuer die verifizierte Multi-Source-Baseline:
  - `docs/BA_MVP_Stage3_Baseline.md`
- Aktualisierte Doku:
  - `docs/MVP_Training_Plan.md`
  - `docs/runbook.md`
  - `docs/Obstacle4_E2E_Results.md`
- `docs/reviews/Codex-Task-Report_last.md` auf diesen Task aktualisiert

## Was wurde wirklich verifiziert?
- GPU-Sichtbarkeit im ModelMaker-Interpreter:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```
  - Exit-Code: `0`
  - Ergebnis:
    - `tensorflow=2.8.4`
    - `GPU:0` sichtbar

- Reale Diagnose des ersten Stage-3-Trainingsfehlers:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python - <<'PY'
from tflite_model_maker import object_detector
train_data, val_data, test_data = object_detector.DataLoader.from_csv(
    filename='work/datasets/ba_mvp_stage3_balanced_multisource/modelmaker.csv',
    images_dir='work/datasets/ba_mvp_stage3_balanced_multisource/images',
)
print('OK', train_data is not None, val_data is not None, test_data is not None)
print(train_data.label_map)
PY
```
  - erster Lauf vor dem Fix:
    - Exit-Code: `1`
    - Fehler:
      - `ValueError: Image format not JPEG`
  - nach dem Fix:
    - Exit-Code: `0`
    - Ergebnis:
      - `DataLoader.from_csv(...)` laedt den Stage-3-CSV-Pfad erfolgreich

- Reale JPEG-Diagnose des materialisierten Stage-3-Datensatzes:
```bash
python - <<'PY'
import json
from pathlib import Path
from PIL import Image
p = Path('work/datasets/ba_mvp_stage3_balanced_multisource/instances_materialized.json')
obj = json.loads(p.read_text())
root = Path('work/datasets/ba_mvp_stage3_balanced_multisource/images')
bad = []
for image in obj['images']:
    with Image.open(root / image['file_name']) as im:
        if im.format != 'JPEG':
            bad.append((image['file_name'], im.format))
print('bad_count', len(bad))
print(bad[:10])
PY
```
  - vor dem Fix:
    - `bad_count=6`
    - alle Ausreisser kamen aus `od_ba_v1`
  - nach dem Fix und Neu-Materialisierung:
    - `bad_count=0`

- Reale Neu-Erzeugung der `OD`- und Stage-3-Artefakte nach dem JPEG-Fix:
```bash
PYTHONPATH=src python -m owli_train dataset import obstacle-dataset \
  --dataset-dir 'data/DataSets/Obstacle Dataset' \
  --out-dir work/datasets/od_ba_v1 \
  --label-map configs/label_maps/obstacle_dataset_to_ba.yaml \
  --mode auto

PYTHONPATH=src python -m owli_train dataset validate \
  --coco work/datasets/od_ba_v1/instances_ba_v1.coco.json \
  --images-dir work/datasets/od_ba_v1/images

PYTHONPATH=src python -m owli_train dataset merge coco \
  --manifest configs/merge_ba_mvp_stage3_balanced_multisource.yaml \
  --out work/datasets/ba_mvp_stage3_balanced_multisource/instances_combined.json

PYTHONPATH=src python -m owli_train dataset split \
  --coco work/datasets/ba_mvp_stage3_balanced_multisource/instances_combined.json \
  --out-dir work/splits/ba_mvp_stage3_balanced_multisource \
  --seed 1337 \
  --ensure-train-class-coverage

rm -rf work/datasets/ba_mvp_stage3_balanced_multisource/images

PYTHONPATH=src python -m owli_train dataset materialize-images \
  --coco work/datasets/ba_mvp_stage3_balanced_multisource/instances_combined.json \
  --merge-manifest configs/merge_ba_mvp_stage3_balanced_multisource.yaml \
  --out-images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --out-coco work/datasets/ba_mvp_stage3_balanced_multisource/instances_materialized.json \
  --mode auto

PYTHONPATH=src python -m owli_train dataset validate \
  --coco work/datasets/ba_mvp_stage3_balanced_multisource/instances_materialized.json \
  --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images

PYTHONPATH=src python -m owli_train dataset export modelmaker-csv \
  --coco work/datasets/ba_mvp_stage3_balanced_multisource/instances_materialized.json \
  --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --splits-json work/splits/ba_mvp_stage3_balanced_multisource/splits.json \
  --out work/datasets/ba_mvp_stage3_balanced_multisource/modelmaker.csv
```
  - Exit-Code: `0`
  - Ergebnisse:
    - `work/datasets/od_ba_v1/instances_ba_v1.coco.json`
    - `work/datasets/ba_mvp_stage3_balanced_multisource/instances_combined.json`
    - `work/datasets/ba_mvp_stage3_balanced_multisource/instances_materialized.json`
    - `work/datasets/ba_mvp_stage3_balanced_multisource/modelmaker.csv`

- Reale Trainingsausfuehrung:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet \
  --config configs/efficientdet_lite2_ba_mvp_stage3.yaml \
  --run-name ba-mvp-stage3-20260308 \
  --require-gpu
```
  - Exit-Code: `0`
  - erzeugte Artefakte:
    - `work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite`
    - `work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/labels.txt`
    - `work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/class_names.json`
    - `work/runs/20260308-183140-ba-mvp-stage3-20260308/mapping_files.json`
  - reales Ergebnis:
    - `20` Epochen abgeschlossen
    - `val_det_loss=0.5496`
    - `val_cls_loss=0.4129`
    - `val_box_loss=0.0027`
    - `val_loss=0.5569`
    - `model.tflite` groesse: `7.1 MB`

- Reale TFLite-Inspektion:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite
```
  - Exit-Code: `0`
  - Ergebnis:
    - `builtin_ops_only=true`
    - BA-v1-Labels in kanonischer Reihenfolge in `labels.txt`

- Reale Split-COCO-Erzeugung fuer den Stage-3-Eval:
```bash
PYTHONPATH=src python -m owli_train dataset split \
  --coco work/datasets/ba_mvp_stage3_balanced_multisource/instances_combined.json \
  --out-dir work/splits/ba_mvp_stage3_balanced_multisource \
  --seed 1337 \
  --ensure-train-class-coverage \
  --write-coco
```
  - Exit-Code: `0`
  - erzeugte Artefakte:
    - `work/splits/ba_mvp_stage3_balanced_multisource/instances_train.json`
    - `work/splits/ba_mvp_stage3_balanced_multisource/instances_val.json`
    - `work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json`

- Reale Stage-3-Eval-Entscheidung:
  - Ein Voll-Eval ueber `4066` Bilder wurde kurz gestartet, aber bewusst abgebrochen.
  - Grund:
    - Laufzeitprojektion fast `1h`
    - enthaelt `TRAIN`-Bilder und ist damit fuer die Baseline weniger sauber
  - Stattdessen wurde der gehaltene `TEST`-Split (`408` Bilder) als primaerer Stage-3-Eval verwendet.

- Reale primaere Stage-3-Evaluation auf dem gehaltenen `TEST`-Split:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json \
  --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 8 \
  --out work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_stage3_test.json
```
  - Exit-Code: `0`
  - Report-Artefakte:
    - `work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_stage3_test.json`
    - `work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_stage3_test.md`
  - Ergebnis:
    - `AP=0.1307`
    - `AP50=0.2325`
    - `AP75=0.1270`
    - `AR100=0.2170`
    - `tp=1447`
    - `fp=5612`
    - `fn=2427`
    - `precision=0.2050`
    - `recall=0.3735`
    - alle `10` BA-v1-Klassen mit nicht-null `tp`

- Reale Vergleichsevaluation des neuen Stage-3-Modells auf `Obstacle4` full:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/datasets/obstacle4/instances_combined.json \
  --images-dir data/raw/obstacle4/extracted \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 16 \
  --out work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_obstacle4.json
```
  - Exit-Code: `0`
  - Report-Artefakte:
    - `work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_obstacle4.json`
    - `work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_obstacle4.md`
  - Ergebnis:
    - `AP=0.2443`
    - `AP50=0.3827`
    - `AP75=0.2548`
    - `AR100=0.3926`
    - `tp=1118`
    - `fp=12475`
    - `fn=794`
    - `precision=0.0822`
    - `recall=0.5847`
  - direkter Vergleich zur alten `Obstacle4`-Referenz:
    - `AP`: `0.0952 -> 0.2443`
    - `AP50`: `0.1897 -> 0.3827`
    - `AP75`: `0.0886 -> 0.2548`
    - `AR100`: `0.1899 -> 0.3926`
    - vorher tote Klassen `person`, `bicycle`, `motorcycle`, `bus`, `truck` liefern jetzt echte Treffer

- Reale Golden-Generierung:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite \
  --image data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg \
  --out work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/golden_obstacle4.json \
  --score-threshold 0.1 \
  --max-results 20 \
  --num-threads 8
```
  - Exit-Code: `0`
  - erzeugtes Artefakt:
    - `work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/golden_obstacle4.json`
  - Ergebnis:
    - `20` Detections
    - Klassenmix:
      - `obstacle_pole=8`
      - `car=6`
      - `obstacle_hole=4`
      - `obstacle_fence=2`

- Nur statisch geprueft:
  - keine Android-/On-Device-Verifikation
  - kein `COCO replay`
  - kein zweiter kombinierter Trainingssweep

## Tests
- Gezielter Test fuer den neuen JPEG-Fix:
```bash
python -m pytest tests/test_dataset_import_obstacle_dataset.py
```
  - Exit-Code: `0`
  - Resultat: `2 passed`

- Repo-Pflichtchecks auf dem finalen Stand:
```bash
python -m ruff format .
python -m ruff check .
python -m pytest
```
  - Exit-Codes:
    - `ruff format`: `0`
    - `ruff check`: `0`
    - `pytest`: `0`
  - Resultate:
    - `ruff format`: keine verbleibenden Formatierungsprobleme
    - `ruff check`: alle Checks bestanden
    - `pytest`: kompletter Testlauf bestanden

## Relevante Run-Kommandos
- Stage-3-Trainingsconfig:
  - `configs/efficientdet_lite2_ba_mvp_stage3.yaml`
- Kanonische BA-v1-Label-Map fuer den Lite2-Lauf:
  - `configs/label_contracts/ba_v1.class_names.json`

```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet \
  --config configs/efficientdet_lite2_ba_mvp_stage3.yaml \
  --run-name ba-mvp-stage3-20260308 \
  --require-gpu

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json \
  --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 8 \
  --out work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_stage3_test.json

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/datasets/obstacle4/instances_combined.json \
  --images-dir data/raw/obstacle4/extracted \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 16 \
  --out work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_obstacle4.json

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite \
  --image data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg \
  --out work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/golden_obstacle4.json \
  --score-threshold 0.1 \
  --max-results 20 \
  --num-threads 8
```

## Offene Risiken
- `COCO replay` fehlt noch; die neue Baseline zeigt klar bessere Rehearsal-Signale, aber die Klassen bleiben nicht stabil genug.
- Die False-Positive-Last bleibt hoch, besonders bei den BA-Core-Hindernisklassen.
- `obstacle_pole` bleibt trotz viel Daten schwierig und verrauscht.
- Die Vergleichsauswertung gegen `Obstacle4` full ist fuer Kontinuitaet nuetzlich, aber nicht identisch zum gehaltenen Stage-3-TEST-Protokoll.
- Android-/On-Device-Verhalten wurde in diesem Task nicht verifiziert.

## Nächster sinnvoller Schritt
- Baue und merge ein kleines explizites `COCO replay`-Subset fuer `person`, `bicycle`, `motorcycle`, `car`, `bus` und `truck`, dann trainiere eine Stage-4-Baseline gegen denselben Stage-3-Evalrahmen.
