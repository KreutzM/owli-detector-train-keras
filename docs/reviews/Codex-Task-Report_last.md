# Codex Task Report

## Ziel
- Einen kleinen, reproduzierbaren Crop-Datenpfad fuer kleine BA-Core-Objekte aufbauen, ohne den bestehenden Stage-3-Vollbildpfad zu ersetzen.
- Stage-3 als aktuelle Baseline beibehalten und einen klaren `Stage-3-plus-crops`-Vergleichszweig vorbereiten.
- Den Crop-Exporter, die Zielartefakte und die Anschluss-Pfade real verifizieren.

## Was wurde geändert?
- Neuen YAML-gesteuerten COCO-Crop-Exporter fuer kleine BA-Core-Objekte implementiert:
  - `src/owli_train/data/coco_crops.py`
- Neue CLI fuer den Exportpfad ergaenzt:
  - `src/owli_train/cli.py`
- Reale Crop-Config fuer den Stage-3-Train-Split ergaenzt:
  - `configs/crop_ba_mvp_stage3_small_obstacles.yaml`
- Merge-Manifest und Training-Config fuer den naechsten `Stage-3-plus-crops`-Vergleichslauf ergaenzt:
  - `configs/merge_ba_mvp_stage3_plus_crops.yaml`
  - `configs/efficientdet_lite2_ba_mvp_stage3_plus_crops.yaml`
- Gezielte Tests fuer Crop-Transform, Filter/Caps, Dedupe und CLI-Pfad ergaenzt:
  - `tests/test_dataset_export_coco_crops.py`
  - `tests/test_mvp_data_prep.py`
- Minimale Doku fuer den Crop-Zweig und die Run-Kommandos aktualisiert:
  - `docs/BA_MVP_Stage3_Crops.md`
  - `docs/MVP_Training_Plan.md`
  - `docs/BA_MVP_Stage3_Baseline.md`
  - `docs/runbook.md`
- Reale Zielartefakte erzeugt:
  - `work/datasets/ba_mvp_stage3_crops`
  - `work/datasets/ba_mvp_stage3_plus_crops`

## Was wurde wirklich verifiziert?
- Neue Tests fuer den Crop-Pfad real ausgefuehrt:
```bash
python -m pytest tests/test_dataset_export_coco_crops.py tests/test_mvp_data_prep.py
```
  - Exit-Code: `0`
  - Ergebnis:
    - `14 passed`
    - Crop-Box-Umrechnung stimmt
    - partielle Restboxen werden gefiltert
    - same-class-Duplikate im Crop werden dedupliziert
    - Config-/Manifest-Pfade bleiben konsistent

- Gezielte Ruff-Pruefung auf den geaenderten Crop-Dateien real ausgefuehrt:
```bash
python -m ruff check src/owli_train/data/coco_crops.py src/owli_train/cli.py tests/test_dataset_export_coco_crops.py tests/test_mvp_data_prep.py
```
  - Exit-Code: `0`
  - Ergebnis:
    - alle neuen Crop-Dateien bestehen Ruff

- Realer Crop-Export auf dem Stage-3-`TRAIN`-Split ausgefuehrt:
```bash
PYTHONPATH=src python -m owli_train dataset export coco-crops --config configs/crop_ba_mvp_stage3_small_obstacles.yaml
```
  - Exit-Code: `0`
  - Ergebnis:
    - Export geschrieben nach `work/datasets/ba_mvp_stage3_crops`
    - `528` Crop-Bilder
    - `3001` Annotationen
    - `10` Kategorien
    - Zielverteilung:
      - `obstacle_bump`: `3`
      - `obstacle_fence`: `176`
      - `obstacle_hole`: `149`
      - `obstacle_pole`: `200`
    - Quellverteilung:
      - `obstacle4`: `41`
      - `mapillary_vistas`: `487`
      - `od_ba_v1`: `0`

- Crop-COCO real validiert:
```bash
PYTHONPATH=src python -m owli_train dataset validate --coco work/datasets/ba_mvp_stage3_crops/instances_ba_v1.coco.json --images-dir work/datasets/ba_mvp_stage3_crops/images
```
  - Exit-Code: `0`
  - Ergebnis:
    - `OK COCO: images=528, ann=3001, cats=10`

- Crop-CSV fuer den spaeteren Trainingseinsatz real exportiert:
```bash
PYTHONPATH=src python -m owli_train dataset export modelmaker-csv --coco work/datasets/ba_mvp_stage3_crops/instances_ba_v1.coco.json --images-dir work/datasets/ba_mvp_stage3_crops/images --out work/datasets/ba_mvp_stage3_crops/modelmaker.csv
```
  - Exit-Code: `0`
  - Ergebnis:
    - `3001` CSV-Zeilen
    - `work/datasets/ba_mvp_stage3_crops/modelmaker.csv`

- Unabhaengige Box-Pruefung gegen den Stage-3-Train-COCO real ausgefuehrt:
```bash
python - <<'PY'
# alle Crop-Annotationen gegen instances_train.json und die gespeicherten crop_box-Ausschnitte nachgerechnet
PY
```
  - Exit-Code: `0`
  - Ergebnis:
    - `3001` Crop-Annotationen nachgerechnet
    - `0` Box-Fehler

- `Stage-3-plus-crops`-Merge real erzeugt:
```bash
PYTHONPATH=src python -m owli_train dataset merge coco --manifest configs/merge_ba_mvp_stage3_plus_crops.yaml --out work/datasets/ba_mvp_stage3_plus_crops/instances_combined.json --report-out work/datasets/ba_mvp_stage3_plus_crops/instances_combined.report.json
```
  - Exit-Code: `0`
  - Ergebnis:
    - `4594` Bilder
    - `41400` Annotationen
    - `10` Kategorien
    - keine verbleibenden Merge-Drops:
      - `duplicate_gt_same_class: 0`
      - alle anderen Drop-Zaehler `0`

- `Stage-3-plus-crops`-Images real materialisiert:
```bash
PYTHONPATH=src python -m owli_train dataset materialize-images --coco work/datasets/ba_mvp_stage3_plus_crops/instances_combined.json --merge-manifest configs/merge_ba_mvp_stage3_plus_crops.yaml --out-images-dir work/datasets/ba_mvp_stage3_plus_crops/images --out-coco work/datasets/ba_mvp_stage3_plus_crops/instances_materialized.json --mode auto
```
  - Exit-Code: `0`
  - Ergebnis:
    - initialer Lauf: `4594` Symlinks geschrieben
    - finaler Lauf nach Dedupe-Fix: COCO aktualisiert, Bilder unveraendert wiederverwendet

- Materialisierten `Stage-3-plus-crops`-Datensatz real validiert:
```bash
PYTHONPATH=src python -m owli_train dataset validate --coco work/datasets/ba_mvp_stage3_plus_crops/instances_materialized.json --images-dir work/datasets/ba_mvp_stage3_plus_crops/images
```
  - Exit-Code: `0`
  - Ergebnis:
    - `OK COCO: images=4594, ann=41400, cats=10`

- Kombiniertes Model-Maker-CSV fuer `Stage-3-plus-crops` real geschrieben:
```bash
python - <<'PY'
from pathlib import Path
out = Path('work/datasets/ba_mvp_stage3_plus_crops')
out.mkdir(parents=True, exist_ok=True)
base_csv = Path('work/datasets/ba_mvp_stage3_balanced_multisource/modelmaker.csv').read_text(encoding='utf-8')
crop_csv = Path('work/datasets/ba_mvp_stage3_crops/modelmaker.csv').read_text(encoding='utf-8')
(out / 'modelmaker.csv').write_text(base_csv + crop_csv, encoding='utf-8')
(out / 'modelmaker.class_names.json').write_text(
    Path('work/datasets/ba_mvp_stage3_balanced_multisource/modelmaker.class_names.json').read_text(encoding='utf-8'),
    encoding='utf-8',
)
print(sum(1 for _ in (out / 'modelmaker.csv').open('r', encoding='utf-8')))
PY
```
  - Exit-Code: `0`
  - Ergebnis:
    - `work/datasets/ba_mvp_stage3_plus_crops/modelmaker.csv`
    - `41400` CSV-Zeilen

- Repo-weite Pflicht-Checks real ausgefuehrt:
```bash
python -m ruff format .
python -m ruff check .
python -m pytest
```
  - Exit-Codes: alle `0`
  - Ergebnis:
    - `ruff format .`: `2 files reformatted`
    - `ruff check .`: erfolgreich
    - `pytest`: `149 passed, 5 skipped`

- Nur statisch geprueft:
  - keine neue Lite2-Trainingsausfuehrung mit `configs/efficientdet_lite2_ba_mvp_stage3_plus_crops.yaml`
  - keine neue TFLite-Eval gegen `Stage-3` oder `Stage-3-plus-crops`
  - die Doku-Updates selbst wurden nur inhaltlich/statisch geprueft

## Tests
- Gezielte Crop-Tests:
  - `python -m pytest tests/test_dataset_export_coco_crops.py tests/test_mvp_data_prep.py`
  - Exit-Code: `0`
  - Resultat: `14 passed`
- Gezielte Ruff-Pruefung:
  - `python -m ruff check src/owli_train/data/coco_crops.py src/owli_train/cli.py tests/test_dataset_export_coco_crops.py tests/test_mvp_data_prep.py`
  - Exit-Code: `0`
- Repo-weite Pflicht-Checks:
  - `python -m ruff format .`
  - `python -m ruff check .`
  - `python -m pytest`
  - Exit-Codes: alle `0`
  - Resultat: `149 passed, 5 skipped`

## Relevante Run-Kommandos
- Crop-Export:
```bash
PYTHONPATH=src python -m owli_train dataset export coco-crops --config configs/crop_ba_mvp_stage3_small_obstacles.yaml
```
- Crop-Validate:
```bash
PYTHONPATH=src python -m owli_train dataset validate --coco work/datasets/ba_mvp_stage3_crops/instances_ba_v1.coco.json --images-dir work/datasets/ba_mvp_stage3_crops/images
```
- Crop-CSV:
```bash
PYTHONPATH=src python -m owli_train dataset export modelmaker-csv --coco work/datasets/ba_mvp_stage3_crops/instances_ba_v1.coco.json --images-dir work/datasets/ba_mvp_stage3_crops/images --out work/datasets/ba_mvp_stage3_crops/modelmaker.csv
```
- `Stage-3-plus-crops`-Merge:
```bash
PYTHONPATH=src python -m owli_train dataset merge coco --manifest configs/merge_ba_mvp_stage3_plus_crops.yaml --out work/datasets/ba_mvp_stage3_plus_crops/instances_combined.json --report-out work/datasets/ba_mvp_stage3_plus_crops/instances_combined.report.json
```
- `Stage-3-plus-crops`-Materialize:
```bash
PYTHONPATH=src python -m owli_train dataset materialize-images --coco work/datasets/ba_mvp_stage3_plus_crops/instances_combined.json --merge-manifest configs/merge_ba_mvp_stage3_plus_crops.yaml --out-images-dir work/datasets/ba_mvp_stage3_plus_crops/images --out-coco work/datasets/ba_mvp_stage3_plus_crops/instances_materialized.json --mode auto
```
- `Stage-3-plus-crops`-Validate:
```bash
PYTHONPATH=src python -m owli_train dataset validate --coco work/datasets/ba_mvp_stage3_plus_crops/instances_materialized.json --images-dir work/datasets/ba_mvp_stage3_plus_crops/images
```

## Offene Risiken
- Der neue Pfad ist als Datenzweig vorbereitet, aber noch nicht ueber einen echten Lite2-Trainingslauf gegen Stage-3 evaluiert.
- Die erste feste Heuristik zieht fast alle Crop-Samen aus `Mapillary`; `OD` bleibt unter der aktuellen Prioritaet und den Caps faktisch ungenutzt.
- `obstacle_bump` bleibt im aktuellen Stage-3-Train-Bestand auch nach Crop-Heuristik sehr duenn (`3` Crop-Ziele).
- Das kombinierte `Stage-3-plus-crops`-CSV wird bewusst aus dem bestehenden Stage-3-CSV plus Crop-CSV zusammengesetzt; dieser Pfad ist real geschrieben, aber noch nicht durch ein anschliessendes Training verifiziert.

## Nächster sinnvoller Schritt
- Starte genau einen echten `EfficientDet-Lite2`-Vergleichslauf mit `configs/efficientdet_lite2_ba_mvp_stage3_plus_crops.yaml` und evaluiere ihn direkt gegen denselben Stage-3-`TEST`-Split wie die aktuelle Baseline.
