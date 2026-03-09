# Codex Task Report

## Ziel
- Den naechsten produktnahen Schritt auf Basis der bevorzugten `Stage-3`-Baseline ausfuehren, statt einen weiteren Trainingszweig zu starten.
- Einen real geprueften Stage-3-Betriebspunkt fuer lokale TFLite-Checks festziehen.
- Eine kleine deterministische Akzeptanzsuite fuer BA-Core-Hard-Cases und BA-Core-Hard-Negatives aufbauen und real gegen das vorhandene Stage-3-TFLite-Modell pruefen.

## Was wurde geändert?
- Neue kleine Akzeptanzsuite fuer den Stage-3-Produkt-Gate-Pfad ergaenzt:
  - `configs/acceptance/ba_mvp_stage3_rc01_suite.json`
- Neue Doku fuer den Stage-3-Produkt-Gate-Schritt ergaenzt:
  - `docs/BA_MVP_Stage3_Product_Gate.md`
- Verweise auf den neuen Produkt-Gate-Schritt in bestehenden MVP-Dokus aktualisiert:
  - `docs/MVP_Training_Plan.md`
  - `docs/BA_MVP_Stage3_Baseline.md`
- Pflichtreport auf die real ausgefuehrte Threshold-Kalibrierung und Suite-Verifikation umgestellt:
  - `docs/reviews/Codex-Task-Report_last.md`
- Keine Codeaenderung:
  - kein neuer Runtime-Code
  - keine Android-Aenderung
  - kein neuer Trainingslauf

## Was wurde wirklich verifiziert?
- Reale Threshold-Kalibrierung auf dem identischen Stage-3-`TEST`-Split fuer das bestehende Stage-3-TFLite-Modell ausgefuehrt:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json \
  --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite \
  --score-threshold 0.15 \
  --noise-thresholds 0.1,0.15,0.2,0.25,0.3 \
  --num-threads 8 \
  --out work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_stage3_test_thresh015.json

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json \
  --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite \
  --score-threshold 0.2 \
  --noise-thresholds 0.1,0.2,0.3,0.4 \
  --num-threads 8 \
  --out work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_stage3_test_thresh020.json

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json \
  --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite \
  --score-threshold 0.25 \
  --noise-thresholds 0.1,0.2,0.25,0.3,0.4 \
  --num-threads 8 \
  --out work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_stage3_test_thresh025.json

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json \
  --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite \
  --score-threshold 0.3 \
  --noise-thresholds 0.1,0.2,0.25,0.3,0.4 \
  --num-threads 8 \
  --out work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_stage3_test_thresh030.json
```
  - Exit-Codes: alle `0`
  - Ergebnis:
    - Threshold `0.10` bestehender Referenzpunkt:
      - precision `0.2050`
      - recall `0.3735`
      - FP/100 `1375.49`
    - Threshold `0.15`:
      - precision `0.3188`
      - recall `0.3456`
      - FP/100 `701.23`
    - Threshold `0.20`:
      - precision `0.4596`
      - recall `0.3036`
      - FP/100 `338.97`
    - Threshold `0.25`:
      - precision `0.5672`
      - recall `0.2788`
      - FP/100 `201.96`
    - Threshold `0.30`:
      - precision `0.6829`
      - recall `0.2530`
      - FP/100 `111.52`

- Die neue Akzeptanzsuite gegen den real vorhandenen Stage-3-TFLite-Pfad geprueft:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python - <<'PY'
# lädt configs/acceptance/ba_mvp_stage3_rc01_suite.json und prüft das Stage-3-TFLite-Modell
# auf der 13-Bilder-Suite bei 0.1, 0.15, 0.2 und 0.25;
# schreibt:
# work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/acceptance_stage3_rc01_multi_threshold.json
PY
```
  - Exit-Code: `0`
  - Ergebnis:
    - Suite-Groesse:
      - `13` Bilder
      - `8` BA-Core-Positives
      - `5` BA-Core-Hard-Negatives auf Rehearsal-only-Bildern
    - Multi-threshold Ergebnis:
      - `0.10`: `4/8` Focus-Hits, `2/5` Negatives BA-core-clean
      - `0.15`: `3/8` Focus-Hits, `3/5` Negatives BA-core-clean
      - `0.20`: `2/8` Focus-Hits, `4/5` Negatives BA-core-clean
      - `0.25`: `2/8` Focus-Hits, `5/5` Negatives BA-core-clean
    - Lesart:
      - `0.10` trifft mehr harte BA-Core-Faelle, verschmutzt aber zu viele Hard-Negatives
      - `0.20+` saeubert die Hard-Negatives, verliert aber zu viele harte BA-Core-Faelle
      - `0.15` ist der ehrlichste Arbeits-Kompromiss fuer lokale Produkt-Checks, aber kein Release-Signal

- Die neue Akzeptanzsuite-Datei selbst indirekt real geprueft:
```bash
python - <<'PY'
import json
from pathlib import Path
json.loads(Path('configs/acceptance/ba_mvp_stage3_rc01_suite.json').read_text())
PY
```
  - Exit-Code: `0`
  - Ergebnis:
    - JSON ist gueltig
    - alle im Suite-Lauf referenzierten Bilder waren lokal lesbar

- Nur statisch geprueft:
  - die neue Produkt-Gate-Doku
  - die Verweise in `docs/MVP_Training_Plan.md` und `docs/BA_MVP_Stage3_Baseline.md`
  - kein neuer Trainingslauf
  - kein neuer Android-/Inference-Code

## Tests
- In diesem Task wurden keine Code-Dateien geaendert.
- Deshalb wurden keine neuen `ruff`- oder `pytest`-Laeufe gestartet.
- Die reale Verifikation bestand aus:
  - vier echten TFLite-Eval-Laeufen auf dem Stage-3-`TEST`-Split
  - einem echten Multi-threshold-Akzeptanzlauf auf der neuen 13-Bilder-Suite

## Relevante Run-Kommandos
- Threshold-Kalibrierung:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json \
  --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --model work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite \
  --score-threshold 0.15 \
  --noise-thresholds 0.1,0.15,0.2,0.25,0.3 \
  --num-threads 8 \
  --out work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/eval_efficientdet_tflite_stage3_test_thresh015.json
```
- Suite-Check:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python - <<'PY'
# lädt configs/acceptance/ba_mvp_stage3_rc01_suite.json und schreibt
# work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/acceptance_stage3_rc01_multi_threshold.json
PY
```

## Offene Risiken
- `Stage-3` bleibt die beste Trainingsbaseline, aber der aktuelle TFLite-Pfad ist nach diesem Produkt-Gate noch nicht release-ready.
- Auch beim aktuell ehrlichsten Arbeits-Threshold `0.15` bleiben die harten BA-Core-Faelle zu schwach:
  - nur `3/8` Focus-Hits in der kleinen Hard-Case-Suite
- `0.20+` wuerde die Hard-Negatives sauberer machen, verliert aber noch mehr harte BA-Core-Faelle.
- Ein echter produktiver Einsatzpunkt ist damit weiterhin offen; es gibt im Moment nur einen dokumentierten Arbeits-Betriebspunkt fuer lokale Checks.

## Nächster sinnvoller Schritt
- Starte als naechsten kleinen Produkt-Schritt genau eine Fehlerbild-getriebene Stage-3-Nachbesserung, die nur auf die im Produkt-Gate auffaelligen BA-Core-Hard-Cases zielt (`obstacle_bump`, `obstacle_fence`, `obstacle_hole`), und pruefe danach denselben Threshold-Kalibrierungs- und Suite-Pfad erneut.
