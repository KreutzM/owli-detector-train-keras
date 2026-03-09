# Codex Task Report

## Ziel
- Den ersten echten BA-v2-`EfficientDet-Lite2` / Model-Maker-Vergleichslauf mit der jetzt verfuegbaren kleinen Online-Augmentation ausfuehren.
- Die neue Augmentierungsvariante ehrlich gegen die aktuelle BA-v2-MVP-Baseline ohne Augmentierung vergleichen.
- Die reale Entscheidung dokumentieren, ob `rand_hflip + moderates jitter` als neuer Default im bestehenden BA-v2-MVP-Pfad taugt.

## Was wurde geändert?
- Neue BA-v2-MVP-Augmentierungs-Config fuer den ersten echten Vergleichslauf genutzt:
  - `configs/efficientdet_lite2_ba_v2_mvp_aug.yaml`
  - nur `rand_hflip: true`, `jitter_min: 0.9`, `jitter_max: 1.1`
- Neue Vergleichsdoku fuer den real ausgefuehrten Augmentierungs-Lauf ergaenzt:
  - `docs/BA_v2_MVP_Augmentation_Baseline.md`
- BA-v2-Basisdoku minimal auf den Vergleichslauf verlinkt:
  - `docs/BA_v2_MVP_Baseline.md`
- MVP-Plan um den echten Augmentierungs-Vergleichsstand erweitert:
  - `docs/MVP_Training_Plan.md`
- Runbook um die verifizierten WSL2-Kommandos fuer den Augmentierungs-Lauf erweitert:
  - `docs/runbook.md`
- Pflicht-Report fuer diesen Task aktualisiert:
  - `docs/reviews/Codex-Task-Report_last.md`

## Was wurde wirklich verifiziert?
- Real ausgefuehrt:
  - echter BA-v2-MVP-Trainingslauf mit Online-Augmentation bis zum TFLite-Export
  - `inspect tflite` auf dem exportierten Modell
  - `eval efficientdet-tflite` auf dem BA-v2-`TEST`-Split
  - `golden detect` auf derselben Bildprobe wie in der BA-v2-Baseline
  - JSON-basierter Direktvergleich Baseline vs. Augmentierungs-Lauf fuer globale Metriken, Per-Class-Counts und FP-Last
  - erzeugte Artefakte:
    - `work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/artifacts/model.tflite`
    - `work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/artifacts/labels.txt`
    - `work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/reports/eval_efficientdet_tflite_ba_v2_test.json`
    - `work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/reports/eval_efficientdet_tflite_ba_v2_test.md`
    - `work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/reports/golden_ba_v2_test_mix.json`
- Statisch geprueft:
  - `docs/BA_v2_Hazard_Labelset.md`
  - `docs/BA_v2_MVP_Train_Candidate.md`
  - `docs/BA_v2_MVP_Baseline.md`
  - `docs/MVP_Training_Plan.md`
  - `docs/runbook.md`
  - `docs/BA_v2_Augmentation_Feasibility.md`
  - `configs/efficientdet_lite2_ba_v2_mvp.yaml`
  - `configs/efficientdet_lite2_ba_v2_mvp_aug.yaml`
  - `src/owli_train/training/modelmaker_efficientdet.py`
  - vorheriger Stand von `docs/reviews/Codex-Task-Report_last.md`
- Inhaltlich verifiziert:
  - der Vergleich bleibt fair:
    - gleicher BA-v2-MVP-Datensatz
    - gleicher Split
    - gleicher Lite2-Produktpfad
    - gleiche Epochenzahl
    - gleiche Batchgroesse
    - kein `autoaugment_policy`
  - die erste reale `rand_hflip + jitter`-Variante verbessert nur Teilaspekte einzelner Hazard-Core-Klassen und schlaegt die aktuelle BA-v2-Baseline global nicht

## Tests
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet --config configs/efficientdet_lite2_ba_v2_mvp_aug.yaml --run-name ba-v2-mvp-augmentation-baseline-20260309 --require-gpu`
  - Exit-Code: `0`
  - Ergebnis: echter BA-v2-MVP-Augmentierungs-Lauf abgeschlossen; TFLite exportiert nach `work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/artifacts/model.tflite`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite --model work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/artifacts/model.tflite`
  - Exit-Code: `0`
  - Ergebnis: builtin-ops-only `true`; erwartete EfficientDet-Lite2-Operatoren inkl. `TFLite_Detection_PostProcess`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite --coco work/splits/ba_v2_hazard_slice02_mapillary_od_ground/instances_test.json --images-dir work/datasets/ba_v2_mvp_candidate/images --model work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/artifacts/model.tflite --score-threshold 0.1 --noise-thresholds 0.05,0.1,0.3 --num-threads 8 --out work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/reports/eval_efficientdet_tflite_ba_v2_test.json`
  - Exit-Code: `0`
  - Ergebnis: Eval-Reports JSON und Markdown geschrieben; globale Metriken `AP 0.1140`, `AP50 0.2107`, `AP75 0.1085`, `AR100 0.1948`, `precision 0.1815`, `recall 0.3434`
- `PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect --model work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/artifacts/model.tflite --image work/datasets/ba_v2_mvp_candidate/images/mapillary_vistas/training/ppvi1a8kNPmFjkS6Lhbnsg.jpg --out work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/reports/golden_ba_v2_test_mix.json --score-threshold 0.1 --max-results 20 --num-threads 8`
  - Exit-Code: `0`
  - Ergebnis: `20` Golden-Detections geschrieben; Klassenmix `person 10`, `car 8`, `truck 2`
- `python -m ruff format .`
  - Exit-Code: `0`
  - Ergebnis: Repo nach Abschluss des Patches formatiert
- `python -m ruff check .`
  - Exit-Code: `0`
  - Ergebnis: alle Checks sauber
- `python -m pytest`
  - Exit-Code: `0`
  - Ergebnis: `188 passed, 5 skipped`

## Relevante Run-Kommandos
- Reale WSL2-Kommandos fuer den Vergleichslauf:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet \
  --config configs/efficientdet_lite2_ba_v2_mvp_aug.yaml \
  --run-name ba-v2-mvp-augmentation-baseline-20260309 \
  --require-gpu

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train inspect tflite \
  --model work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/artifacts/model.tflite

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train eval efficientdet-tflite \
  --coco work/splits/ba_v2_hazard_slice02_mapillary_od_ground/instances_test.json \
  --images-dir work/datasets/ba_v2_mvp_candidate/images \
  --model work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/artifacts/model.tflite \
  --score-threshold 0.1 \
  --noise-thresholds 0.05,0.1,0.3 \
  --num-threads 8 \
  --out work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/reports/eval_efficientdet_tflite_ba_v2_test.json

PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train golden detect \
  --model work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/artifacts/model.tflite \
  --image work/datasets/ba_v2_mvp_candidate/images/mapillary_vistas/training/ppvi1a8kNPmFjkS6Lhbnsg.jpg \
  --out work/runs/20260309-183932-ba-v2-mvp-augmentation-baseline-20260309/reports/golden_ba_v2_test_mix.json \
  --score-threshold 0.1 \
  --max-results 20 \
  --num-threads 8
```

## Offene Risiken
- Es wurde genau eine kleine Augmentierungs-Konfiguration real gegen die Baseline gefahren; daraus folgt noch kein belastbarer Sweep ueber alle sinnvollen Jitter-Staerken.
- Die ersten Resultate zeigen gemischtes Klassenverhalten; insbesondere `obstacle_barrier` und `obstacle_hole_dropoff` profitieren leicht, waehrend `obstacle_ground`, `obstacle_pole`, `car` und `bus` nicht profitieren.
- Die FP-Last verbessert sich nur am strengeren Noise-Threshold `0.30`, verschlechtert sich aber bei `0.05` und `0.10`.
- Die qualitative Golden-Probe bleibt weiter ohne Hazard-Core-Detektionen in den Top-20; der Produktengpass liegt also weiterhin nicht primaer an fehlender technischer Augmentationsunterstuetzung.

## Nächster sinnvoller Schritt
- Fahre genau einen engeren BA-v2-Folgelauf mit schwacherer Online-Augmentierung, bevorzugt `rand_hflip` ohne oder mit kleinerem Jitter, bevor weitere Augmentations-Komplexitaet oder ein groesserer Trainingsumbau diskutiert wird.
