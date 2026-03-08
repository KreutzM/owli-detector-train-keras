# Codex Task Report

## Ziel
- Den bestehenden Mapillary-Konverter so erweitern, dass er neben dem bisherigen `v1.2`-Layout auch den lokal vorliegenden `Map2/v2.0`-Annotationsstand verarbeiten kann.
- Dabei den BA-v1-Contract unveraendert lassen und `v2.0` nur explizit per Schalter aktivieren, statt den bisherigen stabilen `v1.2`-Pfad still zu aendern.

## Was wurde geändert?
- `src/owli_train/data/mapillary_vistas.py`
  - unterstuetzt jetzt `v1.2`- und `v2.0`-Mapillary-Layouts
  - erkennt die feingranulare Kategorienquelle robust:
    - `v1.2`: aus `supercategory`
    - `v2.0`: aus `name`
  - kann `training/v1.2/...`, `validation/v1.2/...`, `training/v2.0/...`, `validation/v2.0/...` aufloesen
  - fuehrt `annotation_version` in den Exportartefakten/QC mit
- `src/owli_train/cli.py`
  - neuer Schalter `--annotation-version auto|v1.2|v2.0`
  - aktueller Standard bleibt konservativ bei `v1.2`, wenn beide Stande vorhanden sind
- `configs/label_maps/mapillary_vistas_to_ba.yaml`
  - fuer `v2.0` erweitert um `human--person--individual -> person`
  - Status/Notes auf `v1.2` und `v2.0` nachgeschaerft
- Tests erweitert:
  - `tests/test_dataset_import_mapillary_vistas.py`
  - `tests/test_mvp_data_prep.py`
  - `tests/test_cli_help.py`
- Doku aktualisiert:
  - `docs/Mapillary_Vistas_Integration.md`
  - `docs/runbook.md`
  - `docs/MVP_Training_Plan.md`
- `docs/reviews/Codex-Task-Report_last.md` auf diesen Folge-Task aktualisiert.

## Was wurde wirklich verifiziert?
- Tatsaechlich ausgefuehrte Analyse-Kommandos:
  - `git status --short --branch`
  - `sed -n '1,260p' src/owli_train/data/mapillary_vistas.py`
  - `sed -n '1,220p' configs/label_maps/mapillary_vistas_to_ba.yaml`
  - Python-Inspektionen gegen:
    - `data/DataSets/Map2/config_v1.2.json`
    - `data/DataSets/Map2/config_v2.0.json`
    - `data/DataSets/Map2/training/v1.2/panoptic/panoptic_2018.json`
    - `data/DataSets/Map2/validation/v1.2/panoptic/panoptic_2018.json`
    - `data/DataSets/Map2/training/v2.0/panoptic/panoptic_2020.json`
    - `data/DataSets/Map2/validation/v2.0/panoptic/panoptic_2020.json`
- Reale Datensatzbefunde:
  - `Map2` enthaelt dieselben Bilder wie `Map`, aber zusaetzlich zwei parallele Annotationsstaende:
    - `v1.2`
    - `v2.0`
  - `v2.0` ist nicht drop-in-identisch zur alten Namensseite
  - fuer BA-v1 relevant:
    - `human--person` wird in `v2.0` zu `human--person--individual`
    - die anderen derzeit gemappten Zielklassen bleiben als feingranulare `name`-Felder vorhanden
- Reale Konvertierung ausgefuehrt:
```bash
python -m owli_train dataset import mapillary-vistas \
  --mapillary-dir data/DataSets/Map2 \
  --out-dir data/processed/mapillary_ba_v2_0_sample \
  --label-map configs/label_maps/mapillary_vistas_to_ba.yaml \
  --annotation-version v2.0 \
  --max-long-side 1600 \
  --limit-images-per-split 100
```
- Reale Nachpruefung des Exports:
```bash
python -m owli_train dataset validate \
  --coco data/processed/mapillary_ba_v2_0_sample/instances_ba_v1.coco.json \
  --images-dir data/processed/mapillary_ba_v2_0_sample/images
```
- Ergebnis:
  - `images=200`
  - `annotations=6535`
  - `cats=9`
- Reale QC-Zahlen aus `data/processed/mapillary_ba_v2_0_sample/qc_report.json`:
  - `annotation_version`: `v2.0`
  - `training`: `100` Bilder, `3315` Annotationen
  - `validation`: `100` Bilder, `3220` Annotationen
  - Zielklassen bleiben:
    - `obstacle_fence`
    - `obstacle_hole`
    - `obstacle_pole`
    - `bicycle`
    - `bus`
    - `car`
    - `motorcycle`
    - `person`
    - `truck`
  - `person` wird im `v2.0`-Pfad jetzt korrekt ueber `human--person--individual` aufgeloest
- Rein statisch geprueft:
  - `object--manhole` bleibt ungemappt
  - rider-Klassen bleiben ungemappt
  - `person-group` bleibt ungemappt
  - Defaultverhalten bleibt bewusst konservativ bei `v1.2`

## Tests
- `python -m pytest tests/test_dataset_import_mapillary_vistas.py tests/test_mvp_data_prep.py tests/test_cli_help.py`
  - Exit-Code: `0`
  - Resultat: `22 passed`
- `python -m ruff format .`
  - Exit-Code: `0`
  - Resultat: wird nach diesem Bericht erneut auf dem Gesamtstand ausgefuehrt
- `python -m ruff check .`
  - Exit-Code: `0`
  - Resultat: wird nach diesem Bericht erneut auf dem Gesamtstand ausgefuehrt
- `python -m pytest`
  - Exit-Code: `0`
  - Resultat: wird nach diesem Bericht erneut auf dem Gesamtstand ausgefuehrt

## Relevante Run-Kommandos
- Bisheriger konservativer Pfad:
```bash
python -m owli_train dataset import mapillary-vistas \
  --mapillary-dir data/DataSets/Map \
  --out-dir data/processed/mapillary_ba_v1 \
  --label-map configs/label_maps/mapillary_vistas_to_ba.yaml \
  --max-long-side 1600
```
- Expliziter `Map2/v2.0`-Pfad:
```bash
python -m owli_train dataset import mapillary-vistas \
  --mapillary-dir data/DataSets/Map2 \
  --out-dir data/processed/mapillary_ba_v2_0 \
  --label-map configs/label_maps/mapillary_vistas_to_ba.yaml \
  --annotation-version v2.0 \
  --max-long-side 1600
```
- Reale verifizierte `v2.0`-Sample-Ausfuehrung:
```bash
python -m owli_train dataset import mapillary-vistas \
  --mapillary-dir data/DataSets/Map2 \
  --out-dir data/processed/mapillary_ba_v2_0_sample \
  --label-map configs/label_maps/mapillary_vistas_to_ba.yaml \
  --annotation-version v2.0 \
  --max-long-side 1600 \
  --limit-images-per-split 100
```

## Offene Risiken
- Die lokale Lizenzdatei weist weiterhin auf `CC BY-NC-SA` hin; das bleibt fuer produktnahe Nutzung separat zu bewerten.
- `v2.0` fuehrt mehr Taxonomiebreite ein; der aktuelle Patch nutzt davon bewusst nur die kleine BA-v1-relevante Teilmenge.
- `object--manhole`, rider-Klassen und `human--person--person-group` bleiben absichtlich ausgeschlossen.
- Der reale `v2.0`-Nachweis ist erneut ein begrenzter Sample-Import, kein Vollimport ueber alle gelabelten Bilder.

## Nächster sinnvoller Schritt
- Den `Map2/v2.0`-Konverter auf einem groesseren oder vollstaendigen Bestand laufen lassen und dann entscheiden, ob der MVP-Merge-Pfad von `Map/v1.2` auf `Map2/v2.0` umgestellt werden soll.
