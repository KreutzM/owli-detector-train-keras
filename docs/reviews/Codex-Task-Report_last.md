# Codex Task Report

## Ziel
- Den langsamen `EfficientDet-TFLite`-Eval-Pfad fuer CPU-Systeme beschleunigen.
- Bild-Inferenz optional ueber mehrere Prozesse parallelisieren, ohne das bestehende Default-Verhalten zu aendern.
- Den neuen Bedienpfad mit kleinem, offline-faehigem Testumfang absichern.

## Was wurde geändert?
- Optionalen Multiprocess-Eval-Pfad fuer `eval efficientdet-tflite` implementiert:
  - `src/owli_train/eval/efficientdet_tflite.py`
- CLI um `--num-workers` erweitert:
  - `src/owli_train/cli.py`
- Leichte Tests fuer Config, CLI und Parallel-Auswahl ergaenzt/aktualisiert:
  - `tests/test_eval_efficientdet_tflite_config.py`
  - `tests/test_eval_efficientdet_tflite_cli.py`
  - `tests/test_eval_efficientdet_tflite_parallel.py`
- Runbook um den neuen Parallel-Pfad ergaenzt:
  - `docs/runbook.md`
- Pflicht-Report fuer diesen Task aktualisiert:
  - `docs/reviews/Codex-Task-Report_last.md`

## Was wurde wirklich verifiziert?
- Relevante Implementierung und bestehende Tests vor dem Patch gelesen:
```bash
sed -n '800,920p' src/owli_train/cli.py
sed -n '1,260p' src/owli_train/eval/efficientdet_tflite.py
sed -n '1,260p' tests/test_eval_efficientdet_tflite_config.py
sed -n '1,260p' tests/test_eval_efficientdet_tflite_cli.py
sed -n '1,260p' tests/test_eval_efficientdet_tflite_empty_detections.py
```
  - Exit-Code: `0`
  - Ergebnis:
    - der bestehende Eval-Pfad war seriell pro Bild
    - `--num-threads` ging nur in einen einzelnen TFLite-Interpreter
    - ein opt-in Multiprocess-Pfad ist der kleinste saubere Beschleunigungshebel

- Gezielte Pytest-Suite fuer den neuen Pfad real ausgefuehrt:
```bash
python -m pytest tests/test_eval_efficientdet_tflite_config.py tests/test_eval_efficientdet_tflite_cli.py tests/test_eval_efficientdet_tflite_empty_detections.py tests/test_eval_efficientdet_tflite_parallel.py
```
  - Exit-Code: `0`
  - Ergebnis:
    - `10 passed`
    - Config-Validierung bleibt intakt
    - CLI reicht `--num-workers` korrekt weiter
    - Eval wechselt bei `num_workers > 1` in den Parallel-Pfad

- Gezielte Ruff-Checks auf den geaenderten Dateien real ausgefuehrt:
```bash
python -m ruff check src/owli_train/eval/efficientdet_tflite.py src/owli_train/cli.py tests/test_eval_efficientdet_tflite_config.py tests/test_eval_efficientdet_tflite_cli.py tests/test_eval_efficientdet_tflite_parallel.py
```
  - Exit-Code: `0`
  - Ergebnis:
    - alle geaenderten Dateien bestehen Ruff

- Vollstaendige Repo-Checks real ausgefuehrt:
```bash
python -m ruff format .
python -m ruff check .
python -m pytest
```
  - Exit-Code: `0`
  - Ergebnis:
    - `ruff format .`: erfolgreich
    - `ruff check .`: erfolgreich
    - `pytest`: erfolgreich

- Realer Smoke-Run des neuen Parallel-Pfads ausgefuehrt:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_mvp_stage4_with_coco_replay/instances_test.json \
  --images-dir work/datasets/ba_mvp_stage4_with_coco_replay/images \
  --model work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite \
  --limit-images 8 \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-workers 2 \
  --num-threads 1 \
  --out work/reports/eval_efficientdet_tflite_stage4_test_parallel_smoke.json
```
  - Exit-Code: `0`
  - Ergebnis:
    - neuer `--num-workers`-Pfad laeuft real durch
    - Report geschrieben:
      - `work/reports/eval_efficientdet_tflite_stage4_test_parallel_smoke.json`
      - `work/reports/eval_efficientdet_tflite_stage4_test_parallel_smoke.md`

- Nur statisch geprueft:
  - keine reale Performance-Messung auf einem grossen Eval-Set im Multiprocess-Modus
  - keine neue GPU-Eval-Implementierung

## Tests
- Gezielte Tests:
  - `python -m pytest tests/test_eval_efficientdet_tflite_config.py tests/test_eval_efficientdet_tflite_cli.py tests/test_eval_efficientdet_tflite_empty_detections.py tests/test_eval_efficientdet_tflite_parallel.py`
  - Exit-Code: `0`
  - Resultat: `10 passed`
- Repo-weite Pflicht-Checks:
  - `python -m ruff format .`
  - `python -m ruff check .`
  - `python -m pytest`
  - Exit-Codes: alle `0`

## Relevante Run-Kommandos
- Seriell wie bisher:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite --coco work/splits/ba_mvp_stage4_with_coco_replay/instances_test.json --images-dir work/datasets/ba_mvp_stage4_with_coco_replay/images --model work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite --num-threads 8
```
- Parallel ueber mehrere Prozesse:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite --coco work/splits/ba_mvp_stage4_with_coco_replay/instances_test.json --images-dir work/datasets/ba_mvp_stage4_with_coco_replay/images --model work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite --num-workers 8 --num-threads 1
```

## Offene Risiken
- Mehr Prozesse helfen nur beim Bild-Sharding; einzelne TFLite-Invokes bleiben CPU-bound.
- Zu hohe Kombinationen aus `--num-workers` und `--num-threads` koennen die CPU uebersubskribieren und sogar bremsen.
- Der neue Pfad nutzt bewusst mehrere CPU-Prozesse, nicht GPU-Delegates; GPU-Eval bleibt weiterhin offen.

## Nächster sinnvoller Schritt
- Miss den neuen Multiprocess-Pfad einmal auf dem realen Stage-4-Testsplit mit `--num-workers` in mehreren Stufen, z. B. `4`, `8`, `12`, bei kleinem `--num-threads`, und halte den besten Durchsatz im Runbook fest.
