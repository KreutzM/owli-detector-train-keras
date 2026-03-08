# Codex Task Report

## Ziel
- Den vorhandenen Mapillary-Vistas-Konverter vom Sample-Status auf einen real nutzbaren BA-v1-Merge-Baustein heben.
- Einen grossen bzw. vollstaendigen `Map/v1.2`-Export real ausfuehren, die Datenqualitaet knapp bewerten und den Export ueber ein konkretes Manifest in den bestehenden MVP-Pfad einhaengen.

## Was wurde geändert?
- Neues konkretes Merge-Manifest:
  - `configs/merge_ba_mvp_stage2_obstacle4_mapillary.yaml`
- Kleiner Konsistenztest:
  - `tests/test_mvp_data_prep.py`
    - prueft, dass das Stage-2-Manifest konkret bleibt und File-Name-Praefixe fuer beide Quellen setzt
- Doku aktualisiert:
  - `docs/Mapillary_Vistas_Integration.md`
  - `docs/MVP_Training_Plan.md`
  - `docs/runbook.md`
- `docs/reviews/Codex-Task-Report_last.md` auf diesen Task aktualisiert

## Was wurde wirklich verifiziert?
- Reale Vollausfuehrung des Mapillary-Exports:
```bash
python -m owli_train dataset import mapillary-vistas \
  --mapillary-dir data/DataSets/Map \
  --out-dir work/datasets/mapillary_vistas_ba_v1 \
  --label-map configs/label_maps/mapillary_vistas_to_ba.yaml \
  --max-long-side 1600
```
  - Exit-Code: `0`
- Reale Export-Ergebnisse:
  - `annotation_version`: `v1.2`
  - `work/datasets/mapillary_vistas_ba_v1/instances_ba_v1.coco.json`
  - `work/datasets/mapillary_vistas_ba_v1/annotations_train.coco.json`
  - `work/datasets/mapillary_vistas_ba_v1/annotations_val.coco.json`
  - `work/datasets/mapillary_vistas_ba_v1/splits.json`
  - `work/datasets/mapillary_vistas_ba_v1/class_names.json`
  - `work/datasets/mapillary_vistas_ba_v1/qc_report.json`
- Reale Validierung des fertigen Exports:
```bash
python -m owli_train dataset validate \
  --coco work/datasets/mapillary_vistas_ba_v1/instances_ba_v1.coco.json \
  --images-dir work/datasets/mapillary_vistas_ba_v1/images
```
  - Exit-Code: `0`
  - Ergebnis: `images=19962`, `ann=629801`, `cats=9`
- Reale QC-Auswertung des fertigen Exports:
  - exportierte Bilder: `19962`
  - exportierte Annotationen: `629801`
  - Kategorien: `9`
  - Bildbaumgroesse: ca. `7.65 GB`
  - `training`
    - gescannte Bilder: `18000`
    - exportierte Bilder: `17966`
    - verworfen nach BA-Filter: `34`
    - Annotationen: `567060`
  - `validation`
    - gescannte Bilder: `2000`
    - exportierte Bilder: `1996`
    - verworfen nach BA-Filter: `4`
    - Annotationen: `62741`
  - kombinierte Klassenhaeufigkeiten:
    - `obstacle_fence`: `12528`
    - `obstacle_hole`: `479`
    - `obstacle_pole`: `383513`
    - `bicycle`: `7155`
    - `bus`: `4910`
    - `car`: `148636`
    - `motorcycle`: `6270`
    - `person`: `58686`
    - `truck`: `7624`
  - kleine Boxen:
    - `min_side < 8`: `207230`
    - `min_side < 16`: `327376`
    - `area < 32^2`: `340540`
  - alle erwarteten Mapillary-Zielklassen kommen vor
  - `obstacle_bump` kommt in Mapillary weiterhin nicht vor
- Reale Merge-Ausfuehrung gegen den bestehenden Obstacle4-Anker:
```bash
python -m owli_train dataset merge coco \
  --manifest configs/merge_ba_mvp_stage2_obstacle4_mapillary.yaml \
  --out work/datasets/ba_mvp_stage2_obstacle4_mapillary/instances_combined.json
```
  - Exit-Code: `0`
  - Ergebnis:
    - `work/datasets/ba_mvp_stage2_obstacle4_mapillary/instances_combined.json`
    - `work/datasets/ba_mvp_stage2_obstacle4_mapillary/instances_combined.report.json`
    - `images=21212`
    - `annotations=631297`
    - `categories=10`
  - Merge-Report:
    - `obstacle4_combined`: `1912 / 1912` Annotationen behalten
    - `mapillary_vistas_ba_v1`: `629385 / 629801` Annotationen behalten
    - `duplicate_gt_same_class`: `416`
- Reale Split-Ausfuehrung mit Coverage-Gate auf dem kombinierten Datensatz:
```bash
python -m owli_train dataset split \
  --coco work/datasets/ba_mvp_stage2_obstacle4_mapillary/instances_combined.json \
  --out-dir work/splits/ba_mvp_stage2_obstacle4_mapillary \
  --seed 1337 \
  --ensure-train-class-coverage
```
  - Exit-Code: `0`
  - Ergebnis: `work/splits/ba_mvp_stage2_obstacle4_mapillary/splits.json`
  - Reale `TRAIN`-Class-Coverage:
    - `missing_train_classes`: `[]`
    - `bus` ist im `TRAIN` enthalten: `3985`
    - alle 10 Klassen sind im `TRAIN` vorhanden
- Nur statisch geprueft:
  - Die spaetere `materialize-images`- und `modelmaker.csv`-Fortsetzung des kombinierten Stage-2-Pfads wurde im Runbook aktualisiert, aber in diesem Task nicht real ausgefuehrt.

## Tests
- `python -m ruff format .`
  - Exit-Code: `0`
  - Resultat: `60 files left unchanged`
- `python -m ruff check .`
  - Exit-Code: `0`
  - Resultat: `All checks passed!`
- `python -m pytest`
  - Exit-Code: `0`
  - Resultat: `130 passed, 5 skipped`

## Relevante Run-Kommandos
```bash
python -m owli_train dataset import mapillary-vistas \
  --mapillary-dir data/DataSets/Map \
  --out-dir work/datasets/mapillary_vistas_ba_v1 \
  --label-map configs/label_maps/mapillary_vistas_to_ba.yaml \
  --max-long-side 1600
```

```bash
python -m owli_train dataset validate \
  --coco work/datasets/mapillary_vistas_ba_v1/instances_ba_v1.coco.json \
  --images-dir work/datasets/mapillary_vistas_ba_v1/images
```

```bash
python -m owli_train dataset merge coco \
  --manifest configs/merge_ba_mvp_stage2_obstacle4_mapillary.yaml \
  --out work/datasets/ba_mvp_stage2_obstacle4_mapillary/instances_combined.json
```

```bash
python -m owli_train dataset split \
  --coco work/datasets/ba_mvp_stage2_obstacle4_mapillary/instances_combined.json \
  --out-dir work/splits/ba_mvp_stage2_obstacle4_mapillary \
  --seed 1337 \
  --ensure-train-class-coverage
```

## Offene Risiken
- Die lokale Lizenzdatei fuer Mapillary weist weiterhin auf `CC BY-NC-SA` hin; das bleibt fuer produktnahe Nutzung separat zu bewerten.
- Der Datensatz ist stark unausgewogen:
  - `obstacle_pole` dominiert sehr stark
  - `car` ist ebenfalls sehr haeufig
  - `obstacle_hole` bleibt sehr schwach
- Viele Boxen sind klein; das kann Training und Eval auf Lite2 spaeter deutlich beeinflussen.
- `obstacle_bump` fehlt in Mapillary weiterhin komplett und muss ueber andere BA-v1-Quellen im Gesamtcontract erhalten bleiben.

## Nächster sinnvoller Schritt
- Den ersten kombinierten Stage-2-Trainingssatz aus `Obstacle4 + Mapillary + bewusst kleinem Sampling/Balancing` materialisieren und daraus den naechsten echten EfficientDet-Lite2-MVP-Lauf bauen.
