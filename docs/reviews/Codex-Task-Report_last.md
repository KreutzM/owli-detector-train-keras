# Codex Task Report

## Ziel
- Den ersten echten `EfficientDet-Lite2`-Trainingslauf auf dem Stage-4-Datensatz mit kleinem `COCO replay` real ausfuehren.
- Den neuen Stage-4-Lauf direkt gegen die verifizierte Stage-3-Baseline vergleichen.
- Die Resultate im Repo-Stil dokumentieren, ohne BA-v1 oder den Produktpfad zu aendern.

## Was wurde geändert?
- Neue Ergebnisdoku fuer den ersten echten Replay-Vergleich:
  - `docs/BA_MVP_Stage4_Baseline.md`
- Stage-4-Pipeline-Doku auf den neuen Ist-Stand gebracht:
  - `docs/BA_MVP_Stage4_Replay_Pipeline.md`
- Plan- und Runbook-Verweise auf den real verifizierten Stage-4-Lauf aktualisiert:
  - `docs/MVP_Training_Plan.md`
  - `docs/runbook.md`
- Pflicht-Report fuer diesen Real-Run aktualisiert:
  - `docs/reviews/Codex-Task-Report_last.md`

## Was wurde wirklich verifiziert?
- Stage-4-Datenbasis und Label-Contract vor dem Lauf geprueft:
```bash
sed -n '1,220p' docs/MVP_Training_Plan.md
sed -n '1,220p' docs/BA_v1_Labelset.md
sed -n '1,260p' docs/runbook.md
sed -n '1,260p' docs/BA_MVP_Stage3_Baseline.md
sed -n '1,280p' docs/BA_MVP_Stage4_Replay_Pipeline.md
sed -n '1,220p' configs/label_contracts/ba_v1.yaml
sed -n '1,240p' configs/efficientdet_lite2_ba_mvp_stage3.yaml
sed -n '1,240p' configs/efficientdet_lite2_ba_mvp_stage4.yaml
sed -n '1,260p' configs/merge_ba_mvp_stage4_with_coco_replay.yaml
sed -n '1,240p' configs/coco_replay_ba_mvp_stage4.yaml
```
  - Exit-Code: `0`
  - Ergebnis:
    - Stage-4-Datensatz und Split liegen real auf Disk
    - Stage-4 nutzt dieselben Lite2-Hyperparameter wie Stage-3
    - die BA-v1-Reihenfolge wird ueber `configs/label_contracts/ba_v1.class_names.json` erzwungen

- GPU-Sichtbarkeit im realen Model-Maker-Interpreter geprueft:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```
  - Exit-Code: `0`
  - Ergebnis:
    - `tensorflow==2.8.4`
    - `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

- Echter Stage-4-Trainingslauf ausgefuehrt:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet \
  --config configs/efficientdet_lite2_ba_mvp_stage4.yaml \
  --run-name ba-mvp-stage4-20260308 \
  --require-gpu
```
  - Exit-Code: `0`
  - Ergebnis:
    - Run dir: `work/runs/20260308-211806-ba-mvp-stage4-20260308`
    - TFLite: `work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite`
    - Labels: `work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/labels.txt`
    - BA-v1-Reihenfolge unveraendert
    - finale Validierungswerte:
      - `val_det_loss=0.6234`
      - `val_cls_loss=0.3775`
      - `val_box_loss=0.0049`
      - `val_loss=0.6309`

- TFLite inspect real ausgefuehrt:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite \
  --model work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite
```
  - Exit-Code: `0`
  - Ergebnis:
    - `builtin_ops_only: true`
    - Operators:
      - `QUANTIZE`
      - `CONV_2D`
      - `DEPTHWISE_CONV_2D`
      - `ADD`
      - `MAX_POOL_2D`
      - `RESIZE_NEAREST_NEIGHBOR`
      - `RESHAPE`
      - `CONCATENATION`
      - `LOGISTIC`
      - `DEQUANTIZE`
      - `TFLite_Detection_PostProcess`

- Stage-4-Modell auf dem nativen Stage-4-Testsplit real evaluiert:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_mvp_stage4_with_coco_replay/instances_test.json \
  --images-dir work/datasets/ba_mvp_stage4_with_coco_replay/images \
  --model work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 8 \
  --out work/runs/20260308-211806-ba-mvp-stage4-20260308/reports/eval_efficientdet_tflite_stage4_test.json
```
  - Exit-Code: `0`
  - Ergebnis:
    - AP `0.1427`
    - AP50 `0.2663`
    - AP75 `0.1389`
    - AR100 `0.2244`
    - precision `0.2527`
    - recall `0.4162`

- Stage-4-Modell auf dem unveraenderten Stage-3-Testsplit real evaluiert:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json \
  --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --model work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 8 \
  --out work/runs/20260308-211806-ba-mvp-stage4-20260308/reports/eval_efficientdet_tflite_stage3_test.json
```
  - Exit-Code: `0`
  - Ergebnis:
    - AP `0.1232`
    - AP50 `0.2170`
    - AP75 `0.1203`
    - AR100 `0.2095`
    - precision `0.2118`
    - recall `0.3627`

- Stage-3-Modell auf dem Stage-4-Testsplit real gegengetestet:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_mvp_stage4_with_coco_replay/instances_test.json \
  --images-dir work/datasets/ba_mvp_stage4_with_coco_replay/images \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 8 \
  --out work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_stage4_test.json
```
  - Exit-Code: `0`
  - Ergebnis:
    - AP `0.1481`
    - AP50 `0.2729`
    - AP75 `0.1446`
    - AR100 `0.2348`
    - precision `0.2434`
    - recall `0.4199`

- Golden detect real ausgefuehrt:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect \
  --model work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite \
  --image data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg \
  --out work/runs/20260308-211806-ba-mvp-stage4-20260308/reports/golden_obstacle4.json \
  --score-threshold 0.1 \
  --max-results 20 \
  --num-threads 8
```
  - Exit-Code: `0`
  - Ergebnis:
    - `14` Detections
    - Klassenmix:
      - `obstacle_pole=7`
      - `obstacle_hole=3`
      - `obstacle_fence=2`
      - `car=2`

- Nur statisch geprueft:
  - die neue Doku gegen die erzeugten Artefakte und die bestehenden Stage-3-Reports gegengelesen
  - keine weiteren Codepfade ausserhalb des real ausgefuehrten Train/Inspect/Eval/Golden-Sets

## Tests
- Keine separaten `ruff`- oder `pytest`-Laeufe nach den Doku-Aenderungen.
- Grund:
  - in diesem Task wurden keine Code- oder Trainingsconfig-Aenderungen benoetigt
  - die reale Verifikation lag auf den produktnahen Kommandos fuer Training, Inspect, Eval und Golden
- Exit-Codes der real ausgefuehrten Produkt-Kommandos:
  - Training: `0`
  - Inspect: `0`
  - Eval Stage-4-Test: `0`
  - Eval Stage-3-Test: `0`
  - Gegen-Eval Stage-3-Modell auf Stage-4-Test: `0`
  - Golden detect: `0`

## Relevante Run-Kommandos
- WSL2 Train:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet --config configs/efficientdet_lite2_ba_mvp_stage4.yaml --run-name ba-mvp-stage4-20260308 --require-gpu
```
- WSL2 Inspect:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite --model work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite
```
- WSL2 Eval Stage-4-Test:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite --coco work/splits/ba_mvp_stage4_with_coco_replay/instances_test.json --images-dir work/datasets/ba_mvp_stage4_with_coco_replay/images --model work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite --score-threshold 0.1 --noise-thresholds 0.05,0.1,0.3 --num-threads 8 --out work/runs/20260308-211806-ba-mvp-stage4-20260308/reports/eval_efficientdet_tflite_stage4_test.json
```
- WSL2 Eval Stage-3-Test:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite --coco work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images --model work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite --score-threshold 0.1 --noise-thresholds 0.05,0.1,0.3 --num-threads 8 --out work/runs/20260308-211806-ba-mvp-stage4-20260308/reports/eval_efficientdet_tflite_stage3_test.json
```
- WSL2 Gegen-Eval:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite --coco work/splits/ba_mvp_stage4_with_coco_replay/instances_test.json --images-dir work/datasets/ba_mvp_stage4_with_coco_replay/images --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite --score-threshold 0.1 --noise-thresholds 0.05,0.1,0.3 --num-threads 8 --out work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_stage4_test.json
```
- WSL2 Golden detect:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect --model work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite --image data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg --out work/runs/20260308-211806-ba-mvp-stage4-20260308/reports/golden_obstacle4.json --score-threshold 0.1 --max-results 20 --num-threads 8
```

## Offene Risiken
- Der erste echte Stage-4-Run ist belastbar genug fuer eine Baseline-Aussage, aber noch nur ein einzelner Replay-Weighting-Punkt.
- Der aktuelle kleine Replay-Baustein senkt einige FP-Zaehler, verschlechtert aber global AP/Recall; das ist negative Evidenz, noch keine vollstaendige Replay-Abschreibung.
- Ohne weiteren kleinen Replay-Gewichtungsversuch bleibt offen, ob das Problem an der Idee `COCO replay` liegt oder an der aktuellen Selektion/Groesse.
- Die aktuelle Produktempfehlung bleibt deshalb konservativ:
  - Stage-3 behalten
  - Stage-4 nicht promoten

## Nächster sinnvoller Schritt
- Fuehre genau einen kleinen Folgeversuch innerhalb derselben sechs Replay-Klassen aus, der nur die Replay-Gewichtung oder Replay-Selektion reduziert, und vergleiche ihn wieder direkt gegen Stage-3 auf demselben Stage-3-Testsplit.
