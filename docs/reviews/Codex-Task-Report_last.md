# Codex Task Report

## Ziel
- Den ersten echten `EfficientDet-Lite2`-Vergleichslauf auf dem vorbereiteten `Stage-3-plus-crops`-Datensatz ausfuehren.
- Das neue Modell sauber gegen die verifizierte `Stage-3`-Baseline auf demselben Stage-3-`TEST`-Split vergleichen.
- Das exportierte TFLite-Modell, die direkte Eval und `golden detect` real verifizieren und die minimal noetige Ergebnisdoku aktualisieren.

## Was wurde geändert?
- Neue Ergebnisdoku fuer den ersten echten Vergleichslauf ergaenzt:
  - `docs/BA_MVP_Stage3_Plus_Crops_Baseline.md`
- Bestehende Doku fuer Plan, Baseline, Crop-Zweig und Run-Kommandos auf den echten Vergleichslauf aktualisiert:
  - `docs/BA_MVP_Stage3_Crops.md`
  - `docs/MVP_Training_Plan.md`
  - `docs/BA_MVP_Stage3_Baseline.md`
  - `docs/runbook.md`
- Pflichtreport auf den real ausgefuehrten Trainings-/Eval-Lauf umgestellt:
  - `docs/reviews/Codex-Task-Report_last.md`
- Keine Code- oder Config-Aenderung war noetig:
  - der vorhandene `Stage-3-plus-crops`-Trainingspfad war fuer den ersten Vergleichslauf bereits ausreichend und fair gegen Stage-3 aufgesetzt

## Was wurde wirklich verifiziert?
- GPU-Sichtbarkeit in der ModelMaker-Umgebung real geprueft:
```bash
.venv-modelmaker-py39/bin/python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```
  - Exit-Code: `0`
  - Ergebnis:
    - `tensorflow==2.8.4`
    - `GPU:0` sichtbar in der WSL2-ModelMaker-Umgebung

- Echter `Stage-3-plus-crops`-Lite2-Trainingslauf real ausgefuehrt:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet \
  --config configs/efficientdet_lite2_ba_mvp_stage3_plus_crops.yaml \
  --run-name ba-mvp-stage3-plus-crops-20260309 \
  --require-gpu
```
  - Exit-Code: `0`
  - Ergebnis:
    - echter Voll-Lauf auf `work/datasets/ba_mvp_stage3_plus_crops/modelmaker.csv`
    - Run dir: `work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309`
    - `20` Checkpoints geschrieben (`ckpt-1` bis `ckpt-20`)
    - TFLite exportiert nach `work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/artifacts/model.tflite`
    - Laufzeit bis TFLite-Export: ca. `2137.6` Sekunden (`35.6` Minuten)

- Exportiertes TFLite-Modell real inspiziert:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite \
  --model work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/artifacts/model.tflite
```
  - Exit-Code: `0`
  - Ergebnis:
    - builtin ops only: `true`
    - BA-v1-Labelreihenfolge blieb erhalten
    - Operatoren bleiben auf dem bekannten Lite2-Pfad:
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

- Neue TFLite-Eval auf dem identischen Stage-3-`TEST`-Split real ausgefuehrt:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json \
  --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --model work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 8 \
  --out work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/reports/eval_efficientdet_tflite_stage3_test.json
```
  - Exit-Code: `0`
  - Ergebnis:
    - `408` Bilder evaluiert
    - AP `0.1280`
    - AP50 `0.2276`
    - AP75 `0.1202`
    - AR100 `0.2142`
    - precision `0.2083`
    - recall `0.3684`
    - FP `5424`
    - FP/100 bei `0.10`: `1329.41`

- Direkten Vergleich gegen die vorhandene Stage-3-Baseline real aus dem Reportmaterial nachgerechnet:
```bash
python - <<'PY'
import json
from pathlib import Path
base = json.loads(Path('work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_stage3_test.json').read_text())
new = json.loads(Path('work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/reports/eval_efficientdet_tflite_stage3_test.json').read_text())
print(base['metrics']['AP'], new['metrics']['AP'])
PY
```
  - Exit-Code: `0`
  - Ergebnis:
    - globale Metriken fallen leicht gegen Stage-3:
      - AP `0.1307 -> 0.1280`
      - AP50 `0.2325 -> 0.2276`
      - AP75 `0.1270 -> 0.1202`
      - AR100 `0.2170 -> 0.2142`
      - recall `0.3735 -> 0.3684`
    - aggregate precision steigt leicht:
      - `0.2050 -> 0.2083`
    - low-threshold FP-Last sinkt leicht:
      - FP bei `0.10`: `5612 -> 5424`
    - auffaellige Per-Class-Änderungen:
      - `obstacle_fence` verbessert sich klar
      - `obstacle_pole` gewinnt etwas recall, aber mit mehr FP
      - `obstacle_hole` senkt FP, verliert aber recall
      - `obstacle_bump` verschlechtert sich

- `golden detect` auf dem bisherigen Referenzbild real ausgefuehrt:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect \
  --model work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/artifacts/model.tflite \
  --image data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg \
  --out work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/reports/golden_obstacle4.json \
  --score-threshold 0.1 \
  --max-results 20 \
  --num-threads 8
```
  - Exit-Code: `0`
  - Ergebnis:
    - `15` Detections geschrieben
    - Klassenmix:
      - `obstacle_pole`: `7`
      - `car`: `3`
      - `obstacle_hole`: `3`
      - `obstacle_fence`: `1`
      - `obstacle_bump`: `1`

- Nur statisch geprueft:
  - keine Code- oder Config-Aenderung war fuer diesen Vergleichslauf noetig
  - die Doku-Aenderungen selbst wurden nur inhaltlich/statisch geprueft
  - es wurde kein zusaetzlicher `Obstacle4`-Full-Eval-Lauf fuer das neue Modell gestartet

## Tests
- In diesem Task wurden keine Codeaenderungen gemacht.
- Deshalb wurden keine neuen `ruff`- oder `pytest`-Laeufe gestartet.
- Reale Verifikation lag in den ausgefuehrten GPU-/Train-/Inspect-/Eval-/Golden-Kommandos.

## Relevante Run-Kommandos
- GPU check:
```bash
.venv-modelmaker-py39/bin/python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```
- Train:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet \
  --config configs/efficientdet_lite2_ba_mvp_stage3_plus_crops.yaml \
  --run-name ba-mvp-stage3-plus-crops-20260309 \
  --require-gpu
```
- Inspect:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite \
  --model work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/artifacts/model.tflite
```
- Eval on the Stage-3 `TEST` split:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json \
  --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --model work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 8 \
  --out work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/reports/eval_efficientdet_tflite_stage3_test.json
```
- Golden:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect \
  --model work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/artifacts/model.tflite \
  --image data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg \
  --out work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/reports/golden_obstacle4.json \
  --score-threshold 0.1 \
  --max-results 20 \
  --num-threads 8
```

## Offene Risiken
- Die globale Metrik bewegt sich leicht nach unten; der aktuelle Crop-Zweig ist damit fachlich kein Baseline-Ersatz.
- `obstacle_bump` bleibt auch mit dem Crop-Zweig zu schwach und verschlechtert sich im ersten echten Vergleichslauf.
- `obstacle_pole` gewinnt etwas recall, aber die FP-Last steigt dort sichtbar an.
- Es wurde in diesem Task kein zusaetzlicher Full-`Obstacle4`-Eval-Lauf fuer das neue Modell gerechnet; die direkte Hauptbewertung basiert bewusst auf dem identischen Stage-3-`TEST`-Split.

## Nächster sinnvoller Schritt
- Behalte `Stage-3` als Hauptbaseline und pruefe als naechsten kleinen Schritt nur eine gezielte zweite Crop-Heuristik, die `obstacle_bump` staerkt und `obstacle_pole`-FP-Druck begrenzt, bevor ein weiterer Voll-Lauf gestartet wird.
