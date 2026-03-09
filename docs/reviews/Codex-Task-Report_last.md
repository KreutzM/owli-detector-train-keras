# Codex Task Report

## Ziel
- Den aktuellen BA-v2-`EfficientDet-Lite2` / Model-Maker-Trainingspfad technisch sauber auf echte Online-Augmentation pruefen.
- Belegen, ob es im bestehenden Produktpfad einen kleinen, reviewbaren Hook fuer CPU-seitige Train-Augmentation mit korrekt mittransformierten Bounding Boxes gibt.
- Wenn sinnvoll, nur den kleinsten Probe-Hook implementieren, ohne den bestehenden Produktpfad umzubauen.

## Was wurde geändert?
- Kleinen optionalen Augmentations-Hook im bestehenden Model-Maker-Wrapper ergaenzt:
  - `src/owli_train/training/modelmaker_efficientdet.py`
  - neue optionale Config `train.augmentation`
  - Uebersetzung auf EfficientDet-`model_spec.hparams`
  - Default-Verhalten bleibt ohne Augmentations-Config unveraendert
- Kleine Offline-Tests fuer Config-Parsing und Hook-Verdrahtung ergaenzt:
  - `tests/test_train_efficientdet_config.py`
  - `tests/test_train_efficientdet_pipeline.py`
- Kleine technische Entscheidungsnotiz ergaenzt:
  - `docs/BA_v2_Augmentation_Feasibility.md`
- MVP-Plan minimal auf die neue Entscheidungsnotiz verlinkt:
  - `docs/MVP_Training_Plan.md`
- Pflicht-Report fuer diesen Task aktualisiert:
  - `docs/reviews/Codex-Task-Report_last.md`

## Was wurde wirklich verifiziert?
- Statisch geprueft:
  - `README.md`
  - `src/owli_train/training/modelmaker_efficientdet.py`
  - `src/owli_train/data/modelmaker_csv.py`
  - `src/owli_train/cli.py`
  - `configs/efficientdet_lite2_ba_v2_mvp.yaml`
  - `docs/runbook.md`
  - `docs/MVP_Training_Plan.md`
  - `docs/BA_v2_MVP_Train_Candidate.md`
  - `docs/BA_v2_MVP_Baseline.md`
  - vorheriger Stand von `docs/reviews/Codex-Task-Report_last.md`
- Installierte, tatsaechlich genutzte Model-Maker-Schnittstelle lokal geprueft:
  - `requirements/modelmaker.txt` pinnt:
    - `tensorflow==2.8.4`
    - `tflite-model-maker==0.4.3`
  - in `.venv-modelmaker-py39` real inspiziert:
    - `object_detector.create(...)`
    - `object_detector.DataLoader.from_csv(...)`
    - EfficientDet-`InputReader`
    - EfficientDet-`autoaugment.py`
- Inhaltlich verifiziert:
  - der aktuelle Repo-Pfad ist:
    - COCO -> Model-Maker-CSV -> `DataLoader.from_csv(...)` -> TFRecord-Cache -> EfficientDet-`InputReader`
  - der kleinste saubere Online-Hook liegt im aktuellen Produktpfad am `model_spec.hparams`
  - Bounding Boxes werden im verwendeten EfficientDet-Dataloader fuer `random_horizontal_flip` sowie fuer Resize/Crop-Jitter gemeinsam mit dem Bild transformiert
  - gezielte photometrische Einzelknobs (`brightness` / `contrast` / `color`) sind im aktuellen Repo-Pfad nicht als saubere Top-Level-Schalter vorhanden; sie sind nur indirekt ueber `autoaugment_policy` erreichbar
  - ein frei gestaltbarer Repo-eigener `tf.data`-Augmentationspfad ist im aktuellen Model-Maker-Pfad nicht minimal-invasiv integrierbar
- Real ausgefuehrt:
  - `python -m ruff format .`
  - `python -m ruff check .`
  - `python -m pytest`
  - zusaetzlich gezielte Offline-Tests auf den betroffenen EfficientDet-Dateien
  - lokale Python-Inspektion der dedizierten `.venv-modelmaker-py39`-API und der installierten Package-Quellen

## Tests
- `python -m ruff format src/owli_train/training/modelmaker_efficientdet.py tests/test_train_efficientdet_config.py tests/test_train_efficientdet_pipeline.py`
  - Exit-Code: `0`
  - Ergebnis: betroffene Dateien formatiert
- `python -m ruff check src/owli_train/training/modelmaker_efficientdet.py tests/test_train_efficientdet_config.py tests/test_train_efficientdet_pipeline.py`
  - Exit-Code: `1` beim ersten Lauf
  - Ergebnis: ein einzelner Annotation-Style-Fix (`UP037`) in `modelmaker_efficientdet.py`
- `python -m pytest tests/test_train_efficientdet_config.py tests/test_train_efficientdet_pipeline.py tests/test_train_efficientdet_cli.py`
  - Exit-Code: `0`
  - Ergebnis: `18 passed`
- `python -m ruff format .`
  - Exit-Code: `0`
  - Ergebnis: Repo nach Abschluss des Patches formatiert
- `python -m ruff check .`
  - Exit-Code: `0`
  - Ergebnis: alle Checks sauber
- `python -m pytest`
  - Exit-Code: `0`
  - Ergebnis: `185 passed, 5 skipped`

## Relevante Run-Kommandos
- Aktuelle BA-v2-Feasibility-Note:
```bash
sed -n '1,260p' docs/BA_v2_Augmentation_Feasibility.md
```
- Optionaler minimaler Online-Augmentations-Hook im bestehenden Produktpfad:
```yaml
train:
  augmentation:
    rand_hflip: true
    jitter_min: 0.9
    jitter_max: 1.1
    autoaugment_policy: v1
```
- WSL2-Trainingskommando im bestehenden Pfad bleibt unveraendert:
```bash
PYTHONPATH=src .venv-modelmaker-py39/bin/python -m owli_train train efficientdet \
  --config configs/efficientdet_lite2_ba_v2_mvp.yaml \
  --require-gpu
```

## Offene Risiken
- Der neue Hook beweist die technische Integrationsstelle, aber nicht den Modellnutzen; es wurde bewusst kein neuer groesserer Trainingslauf gestartet.
- `autoaugment_policy` ist technisch verfuegbar, mischt aber photometrische und geometrische Ops und ist deshalb deutlich weniger kontrollierbar als `rand_hflip` plus moderates `jitter`.
- Eine saubere, frei definierbare Repo-eigene Online-Augmentationspipeline vor Model Maker bleibt ohne groesseren Umbau weiter ausser Reichweite.
- Explizite Translate-only- oder Brightness-only-Schalter sind im aktuellen Produktpfad weiter nicht sauber exponiert.

## Nächster sinnvoller Schritt
- Fahre genau einen kleinen BA-v2-Vergleichslauf im bestehenden Produktpfad mit nur `rand_hflip` plus moderatem `jitter` gegen die aktuelle Baseline, bevor `autoaugment_policy` oder ein groesserer Trainingsumbau ueberhaupt erwogen wird.
