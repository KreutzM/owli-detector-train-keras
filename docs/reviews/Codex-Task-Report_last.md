# Codex Task Report

## Ziel
- Den ersten echten Import-/Konvertierungspfad fuer Mapillary Vistas in das Repo einbauen.
- Aus dem lokal vorhandenen Mapillary-Vistas-Bestand einen BA-gefilterten COCO-Detection-Zwischenstand erzeugen, der zum bestehenden EfficientDet-/ModelMaker-Pfad passt.
- Dabei nur `training` und `validation` verwenden, Bilder auf max. `1600 px` lange Seite verkleinern und nur die bewusst freigegebene BA-/Rehearsal-Klassen-Whitelist uebernehmen.

## Was wurde geändert?
- Neuer Datenkonverter angelegt: `src/owli_train/data/mapillary_vistas.py`
- Neuer CLI-Pfad ergaenzt: `python -m owli_train dataset import mapillary-vistas`
- Reale Mapillary-Label-Map eingefroren: `configs/label_maps/mapillary_vistas_to_ba.yaml`
- Neue Tests ergaenzt:
  - `tests/test_dataset_import_mapillary_vistas.py`
  - `tests/test_mvp_data_prep.py` erweitert
  - `tests/test_cli_help.py` erweitert
- Doku aktualisiert:
  - `docs/Mapillary_Vistas_Integration.md`
  - `docs/MVP_Training_Plan.md`
  - `docs/runbook.md`
- `docs/reviews/Codex-Task-Report_last.md` auf diesen Taskstand aktualisiert.

## Was wurde wirklich verifiziert?
- Tatsaechlich ausgefuehrte Analyse-Kommandos:
  - `sed -n '1,240p' docs/MVP_Training_Plan.md`
  - `sed -n '1,220p' configs/label_maps/mapillary_vistas_to_ba.yaml`
  - `sed -n '1,260p' src/owli_train/cli.py`
  - `sed -n '1,260p' src/owli_train/data/yolo_adapter.py`
  - `sed -n '1,240p' src/owli_train/data/coco.py`
  - `sed -n '1,260p' data/DataSets/Map/demo.py`
  - Python-Inspektionen gegen:
    - `data/DataSets/Map/training/panoptic/panoptic_2018.json`
    - `data/DataSets/Map/validation/panoptic/panoptic_2018.json`
    - `data/DataSets/Map/config.json`
    - Beispielbilder / Beispiel-Instance- und Panoptic-PNGs
- Reale Datensatzbefunde:
  - lokaler Root: `data/DataSets/Map`
  - lokale Edition laut Readme: `Mapillary Vistas Research edition v1.2`
  - lokale Lizenzdatei sagt: `CC BY-NC-SA`
  - `panoptic_2018.json` liefert pro Segment bereits `bbox`, `category_id`, `area` und Bildbezug
  - deshalb wurde `panoptic_2018.json` bewusst als kleinster robuster Detektions-Ableitungspfad gewaehlt
- Reale Konvertierung ausgefuehrt:
```bash
python -m owli_train dataset import mapillary-vistas \
  --mapillary-dir data/DataSets/Map \
  --out-dir data/processed/mapillary_ba_v1_sample \
  --label-map configs/label_maps/mapillary_vistas_to_ba.yaml \
  --max-long-side 1600 \
  --limit-images-per-split 100
```
- Reales Ergebnis dieses Sample-Laufs:
  - `combined COCO`: `data/processed/mapillary_ba_v1_sample/instances_ba_v1.coco.json`
  - `train COCO`: `data/processed/mapillary_ba_v1_sample/annotations_train.coco.json`
  - `val COCO`: `data/processed/mapillary_ba_v1_sample/annotations_val.coco.json`
  - `splits`: `data/processed/mapillary_ba_v1_sample/splits.json`
  - `class_names`: `data/processed/mapillary_ba_v1_sample/class_names.json`
  - `qc report`: `data/processed/mapillary_ba_v1_sample/qc_report.json`
  - `images`: `data/processed/mapillary_ba_v1_sample/images/training/*` und `.../validation/*`
- Reale Nachpruefung des Exports:
```bash
python -m owli_train dataset validate \
  --coco data/processed/mapillary_ba_v1_sample/instances_ba_v1.coco.json \
  --images-dir data/processed/mapillary_ba_v1_sample/images
```
- Ergebnis:
  - `images=200`
  - `annotations=6378`
  - `cats=9`
- Reale QC-Zahlen aus `qc_report.json`:
  - `training`: `100` Bilder, `3228` Annotationen
  - `validation`: `100` Bilder, `3150` Annotationen
  - exportierte Zielklassen:
    - `obstacle_fence`
    - `obstacle_hole`
    - `obstacle_pole`
    - `bicycle`
    - `bus`
    - `car`
    - `motorcycle`
    - `person`
    - `truck`
  - Beispielhafte Resize-Verifikation:
    - `training/--NSVcUgfVhFd6uzkqHOOg.jpg` -> `(1600, 1200)`
    - `validation/--BJs76vloEaiH-wppzWNA.jpg` -> `(1600, 1200)`
- Rein statisch geprueft:
  - `object--manhole` bleibt bewusst ungemappt
  - rider-Klassen bleiben bewusst ungemappt
  - die aktuelle Mapillary-Integration fuehrt keine BA-v1-Vertragsaenderung ein
  - die vom Tasktext vorgeschlagenen Zielnamen `obstacle_fence_rail` und `obstacle_hole_dropoff` wurden absichtlich nicht eingefuehrt, weil der bestehende BA-v1-Contract unveraendert bleiben sollte

## Tests
- `python -m pytest tests/test_dataset_import_mapillary_vistas.py tests/test_mvp_data_prep.py tests/test_cli_help.py`
  - Exit-Code: `0`
  - Resultat: `20 passed`
- `python -m ruff format .`
  - Exit-Code: `0`
  - Resultat: `1 file reformatted, 59 files left unchanged`
- `python -m ruff check . --fix`
  - Exit-Code: `0`
  - Resultat: `1 import-order issue fixed`
- `python -m ruff check .`
  - Exit-Code: `0`
  - Resultat: `All checks passed!`
- `python -m pytest`
  - Exit-Code: `0`
  - Resultat: `127 passed, 5 skipped`

## Relevante Run-Kommandos
- Voller Importpfad:
```bash
python -m owli_train dataset import mapillary-vistas \
  --mapillary-dir data/DataSets/Map \
  --out-dir data/processed/mapillary_ba_v1 \
  --label-map configs/label_maps/mapillary_vistas_to_ba.yaml \
  --max-long-side 1600
```
- Reale verifizierte Sample-Ausfuehrung:
```bash
python -m owli_train dataset import mapillary-vistas \
  --mapillary-dir data/DataSets/Map \
  --out-dir data/processed/mapillary_ba_v1_sample \
  --label-map configs/label_maps/mapillary_vistas_to_ba.yaml \
  --max-long-side 1600 \
  --limit-images-per-split 100
```
- Exportierte Artefakte im verifizierten Sample:
  - `data/processed/mapillary_ba_v1_sample/instances_ba_v1.coco.json`
  - `data/processed/mapillary_ba_v1_sample/annotations_train.coco.json`
  - `data/processed/mapillary_ba_v1_sample/annotations_val.coco.json`
  - `data/processed/mapillary_ba_v1_sample/splits.json`
  - `data/processed/mapillary_ba_v1_sample/class_names.json`
  - `data/processed/mapillary_ba_v1_sample/qc_report.json`

## Offene Risiken
- Die lokale Mapillary-Lizenzdatei weist auf `CC BY-NC-SA` hin; das muss fuer produktnahe Nutzung separat bewusst bewertet werden.
- Der reale Verifikationslauf war ein begrenzter Sample-Import (`100` Bilder pro Split), kein kompletter Vollimport ueber alle `18000 + 2000` gelabelten Bilder.
- `object--manhole` und rider-Klassen bleiben bewusst ungemappt; das ist aktuell Absicht, kann spaeter aber fachlich erneut bewertet werden.
- `object--pothole -> obstacle_hole` ist ein bewusst enger erster Produktentscheid; ob das spaeter fuer die BA-Zielsemantik reicht, muss mit Trainings- und Eval-Ergebnissen belegt werden.

## Nächster sinnvoller Schritt
- Den neuen Mapillary-Konverter auf einem groesseren oder vollstaendigen Bestand laufen lassen und danach den BA-gefilterten Mapillary-Export als echten Merge-Baustein in den MVP-Trainingspfad einhaengen.
