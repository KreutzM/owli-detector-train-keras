# Codex Task Report

## Ziel
- Den `golden detect`-Pfad fuer TFLite-Inference um einen expliziten CPU-Thread-Parameter ergaenzen, damit CPU-Inference nicht still auf einem kleinen Default-Threadpool laeuft, waehrend `eval efficientdet-tflite` bereits `--num-threads` unterstuetzt.

## Was wurde geändert?
- In `src/owli_train/golden/detect.py` wurde `num_threads: int | None` in den Golden-Config-Contract aufgenommen.
- `build_golden_detect_config(...)` validiert jetzt `--num-threads > 0` und speichert den Wert in der Config.
- `generate_golden_detect(...)` reicht `cfg.num_threads` an `create_tflite_runtime(...)` weiter.
- Die CLI `golden detect` in `src/owli_train/cli.py` unterstuetzt jetzt `--num-threads` und delegiert das Flag korrekt an den ModelMaker-Interpreter.
- In `tests/test_golden_config.py`, `tests/test_golden_cli.py` und `tests/test_cli_help.py` wurden gezielte Tests fuer Config-Validierung, Flag-Wiring, Delegation und Help-Ausgabe ergaenzt.
- Die Nutzer-Doku in `docs/android-contract.md` und `docs/runbook.md` wurde knapp um den neuen Performance-Parameter erweitert.
- Geaenderte Dateien:
- `src/owli_train/golden/detect.py`
- `src/owli_train/cli.py`
- `tests/test_golden_config.py`
- `tests/test_golden_cli.py`
- `tests/test_cli_help.py`
- `docs/android-contract.md`
- `docs/runbook.md`

## Was wurde wirklich verifiziert?
- Tatsächlich ausgefuehrte Kommandos:
- `python -m pytest tests/test_golden_config.py tests/test_golden_cli.py tests/test_cli_help.py`
- `python -m ruff format .`
- `python -m ruff check .`
- `python -m pytest`
- Rein statisch geprueft:
- Die eigentliche CPU-Auslastung eines realen Golden-Runs wurde in diesem Task nicht erneut gemessen.
- Es wurde kein echter Benchmark zwischen verschiedenen `--num-threads`-Werten gefahren; der Patch stellt nur die fehlende Steuerbarkeit her.

## Tests
- `python -m pytest tests/test_golden_config.py tests/test_golden_cli.py tests/test_cli_help.py`
- Exit-Code: `0`
- Resultat: `23 passed in 2.25s`.
- `python -m ruff format .`
- Exit-Code: `0`
- Resultat: `56 files left unchanged`.
- `python -m ruff check .`
- Exit-Code: `0`
- Resultat: `All checks passed!`.
- `python -m pytest`
- Exit-Code: `0`
- Resultat: `116 passed, 5 skipped in 1.69s`.

## Relevante Run-Kommandos
- Golden Detect mit expliziter CPU-Threadzahl unter WSL2:
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect --model work/runs/<run_id>/artifacts/model.tflite --image data/raw/obstacle4/extracted/valid/images/-_-_26_005_jpeg.rf.87306b8fa8d39b023b6d8c8354fc529a.jpg --out work/golden/sample.json --score-threshold 0.3 --max-results 20 --num-threads 8`
- Vergleichspfad fuer TFLite-Eval mit denselben Thread-Vorgaben:
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite --coco work/datasets/obstacle4/instances_combined.json --images-dir data/raw/obstacle4/extracted --model work/runs/<run_id>/artifacts/model.tflite --limit-images 100 --score-threshold 0.1 --noise-thresholds 0.1 --num-threads 8`

## Offene Risiken
- `--num-threads` verbessert nur die Steuerbarkeit; ob `8`, `12` oder `16` auf der konkreten CPU am schnellsten ist, bleibt hardware- und Modell-abhaengig.
- `golden detect` bleibt ein Batch-1-Pfad; selbst mit mehr Threads wird die Gesamtauslastung durch Dekodierung, Resize und Python-Postprocessing begrenzt bleiben.
- Ein echter Lauf kann trotz hohem `--num-threads` unter 100 Prozent Gesamt-CPU bleiben, wenn XNNPACK oder der konkrete Op-Mix nicht auf alle Kerne skaliert.

## Nächster sinnvoller Schritt
- Einen kleinen lokalen Throughput-Vergleich fuer `golden detect` oder `eval efficientdet-tflite` mit `--num-threads 4,8,16` auf dem Zielsystem fahren und den besten Default fuer eure CPU dokumentieren.
