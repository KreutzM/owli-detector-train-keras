# Codex Task Report

## Ziel
- Den ersten echten Daten-/Mapping-Schritt auf die neue BA-v2-Hazard-Ontologie umsetzen.
- Einen kleinen, defensiblen, real verifizierten BA-v2-Slice aus den aktuell am besten passenden Quellen erzeugen.
- Den Pfad bis direkt vor einen ersten partiellen BA-v2-Trainingsinput vorbereiten, ohne einen Trainingslauf zu starten.

## Was wurde geändert?
- Kleine Runtime-Erweiterung fuer contract-geordnete Normalisierung ergaenzt:
  - `src/owli_train/data/coco.py`
  - `src/owli_train/cli.py`
- Kleine Runtime-Erweiterung fuer contract-geordneten Merge aus Manifesten ergaenzt:
  - `src/owli_train/data/merge_coco.py`
- Neuer transitorischer Remap von verifizierten BA-v1-Quell-Exporten auf BA-v2 hazard:
  - `configs/label_maps/ba_v1_non_obstacle4_export_to_ba_v2_hazard.yaml`
- Neue BA-v2-Slice-Configs ergaenzt:
  - `configs/balance_ba_v2_hazard_mapillary_slice01.yaml`
  - `configs/balance_ba_v2_hazard_od_slice01.yaml`
  - `configs/merge_ba_v2_hazard_slice01_mapillary_od.yaml`
- Neue dedizierte Slice-Doku ergaenzt:
  - `docs/BA_v2_Hazard_Slice01_Mapillary_OD.md`
- Bestehende Doku minimal auf den ersten realen BA-v2-Slice geschaerft:
  - `docs/MVP_Training_Plan.md`
  - `docs/runbook.md`
- Kleine Tests fuer normalize/merge und BA-v2-Slice-Configs ergaenzt oder erweitert:
  - `tests/test_dataset_normalize.py`
  - `tests/test_dataset_merge_coco.py`
  - `tests/test_ba_v2_mapping_prep.py`

## Was wurde wirklich verifiziert?
- Statisch geprueft:
  - `docs/BA_v1_Labelset.md`
  - `docs/BA_v2_Hazard_Labelset.md`
  - `docs/MVP_Training_Plan.md`
  - `docs/runbook.md`
  - `docs/android-export-contract.md`
  - `configs/label_contracts/ba_v1.yaml`
  - `configs/label_contracts/ba_v2_hazard.yaml`
  - bestehende Label-Maps und Merge-Configs
  - relevante Datenmodule unter `src/owli_train/data/*`
- Inhaltlich verifiziert:
  - der kleinste robuste reale BA-v2-Einstiegspunkt ist aktuell `Mapillary + OD`
  - `Obstacle4` wurde fuer diesen ersten Slice bewusst nicht verwendet
  - `COCO replay` wurde fuer diesen ersten Slice bewusst nicht verwendet
  - der erste reale BA-v2-Slice stuetzt aktuell nur:
    - `obstacle_barrier`
    - `obstacle_hole_dropoff`
    - `obstacle_pole`
  - weiterhin offen bleiben:
    - `obstacle_ground`
    - `obstacle_overhang`
  - der erzeugte Slice ist daher noch kein vollstaendiger BA-v2-Trainingskandidat fuer den finalen Hazard-Contract
- Real ausgefuehrt:
  - contract-geordnete BA-v2-Normalisierung fuer `Mapillary` und `OD`
  - realer BA-v2-Balance-Lauf fuer `Mapillary`
  - realer BA-v2-Pass-through mit QC fuer `OD`
  - realer BA-v2-Merge
  - realer Split mit `--ensure-train-class-coverage`
  - reale Materialisierung der Bilder
  - reale Validierung des materialisierten COCO
  - reale Vorbereitung bis zum `ModelMaker`-CSV

## Tests
- `PYTHONPATH=src python -m owli_train dataset normalize --coco work/datasets/mapillary_vistas_ba_v1/instances_ba_v1.coco.json --images-dir work/datasets/mapillary_vistas_ba_v1/images --label-map configs/label_maps/ba_v1_non_obstacle4_export_to_ba_v2_hazard.yaml --contract configs/label_contracts/ba_v2_hazard.yaml --out work/datasets/mapillary_vistas_ba_v2_hazard_source/instances_normalized.json`
  - Exit-Code: `0`
- `PYTHONPATH=src python -m owli_train dataset normalize --coco work/datasets/od_ba_v1/instances_ba_v1.coco.json --images-dir work/datasets/od_ba_v1/images --label-map configs/label_maps/ba_v1_non_obstacle4_export_to_ba_v2_hazard.yaml --contract configs/label_contracts/ba_v2_hazard.yaml --out work/datasets/od_ba_v2_hazard_source/instances_normalized.json`
  - Exit-Code: `0`
- `PYTHONPATH=src python -m owli_train dataset balance-coco --config configs/balance_ba_v2_hazard_mapillary_slice01.yaml`
  - Exit-Code: `0`
  - Ergebnis: `957` Bilder, `21707` Annotationen, `9` Kategorien
- `PYTHONPATH=src python -m owli_train dataset balance-coco --config configs/balance_ba_v2_hazard_od_slice01.yaml`
  - Exit-Code: `0`
  - Ergebnis: `1592` Bilder, `8911` Annotationen, `7` Kategorien
- `PYTHONPATH=src python -m owli_train dataset merge coco --manifest configs/merge_ba_v2_hazard_slice01_mapillary_od.yaml --out work/datasets/ba_v2_hazard_slice01_mapillary_od/instances_combined.json --report-out work/datasets/ba_v2_hazard_slice01_mapillary_od/instances_combined.report.json`
  - Exit-Code: `0`
  - Ergebnis: `2549` Bilder, `30604` Annotationen, `9` Kategorien
- `PYTHONPATH=src python -m owli_train dataset split --coco work/datasets/ba_v2_hazard_slice01_mapillary_od/instances_combined.json --out-dir work/splits/ba_v2_hazard_slice01_mapillary_od --seed 1337 --ensure-train-class-coverage --write-coco`
  - Exit-Code: `0`
  - Ergebnis: `TRAIN=2039`, `VAL=254`, `TEST=256`
- `PYTHONPATH=src python -m owli_train dataset materialize-images --coco work/datasets/ba_v2_hazard_slice01_mapillary_od/instances_combined.json --merge-manifest configs/merge_ba_v2_hazard_slice01_mapillary_od.yaml --out-images-dir work/datasets/ba_v2_hazard_slice01_mapillary_od/images --out-coco work/datasets/ba_v2_hazard_slice01_mapillary_od/instances_materialized.json --mode auto`
  - Exit-Code: `0`
  - Ergebnis: `2549` Bilder materialisiert, `2549` per Symlink vorhanden
- `PYTHONPATH=src python -m owli_train dataset validate --coco work/datasets/ba_v2_hazard_slice01_mapillary_od/instances_materialized.json --images-dir work/datasets/ba_v2_hazard_slice01_mapillary_od/images`
  - Exit-Code: `0`
  - Ergebnis: `images=2549`, `ann=30604`, `cats=9`
- `PYTHONPATH=src python -m owli_train dataset export modelmaker-csv --coco work/datasets/ba_v2_hazard_slice01_mapillary_od/instances_materialized.json --images-dir work/datasets/ba_v2_hazard_slice01_mapillary_od/images --splits-json work/splits/ba_v2_hazard_slice01_mapillary_od/splits.json --out work/datasets/ba_v2_hazard_slice01_mapillary_od/modelmaker.csv`
  - Exit-Code: `0`
  - Ergebnis: `rows=30604`, `images=2549`, `annotations=30604`
- `python -m ruff format .`
  - Exit-Code: `0`
  - Ergebnis: `1 file reformatted, 66 files left unchanged`
- `python -m ruff check .`
  - Exit-Code: `0`
  - Ergebnis: `All checks passed!`
- `python -m pytest`
  - Exit-Code: `0`
  - Ergebnis: `162 passed, 5 skipped in 5.07s`

## Relevante Run-Kommandos
- BA-v2-Quell-Export-Konvertierung:
```bash
PYTHONPATH=src python -m owli_train dataset normalize \
  --coco work/datasets/mapillary_vistas_ba_v1/instances_ba_v1.coco.json \
  --images-dir work/datasets/mapillary_vistas_ba_v1/images \
  --label-map configs/label_maps/ba_v1_non_obstacle4_export_to_ba_v2_hazard.yaml \
  --contract configs/label_contracts/ba_v2_hazard.yaml \
  --out work/datasets/mapillary_vistas_ba_v2_hazard_source/instances_normalized.json
```
- BA-v2-Slice-Assembly:
```bash
PYTHONPATH=src python -m owli_train dataset merge coco \
  --manifest configs/merge_ba_v2_hazard_slice01_mapillary_od.yaml \
  --out work/datasets/ba_v2_hazard_slice01_mapillary_od/instances_combined.json \
  --report-out work/datasets/ba_v2_hazard_slice01_mapillary_od/instances_combined.report.json
```

## Offene Risiken
- `obstacle_ground` und `obstacle_overhang` bleiben im ersten realen BA-v2-Slice ungestuetzt.
- `obstacle_hole_dropoff` ist vorhanden, aber im Vergleich zu `obstacle_pole` und den Rehearsal-Klassen noch klein.
- `obstacle_barrier` kommt aktuell praktisch nur ueber `Mapillary`.
- Der Slice ist trainingsvorbereitet, aber wegen der zwei fehlenden Hazard-Core-Klassen noch kein vollstaendiger BA-v2-Trainingskandidat fuer den finalen Contract; ein Lite2-Lauf wurde bewusst noch nicht gestartet.
- Bestehende Source-Artefaktdateinamen enthalten in einzelnen Zwischenordnern noch historische `instances_ba_v1.coco.json`-Namen, obwohl der Inhalt fuer diesen Slice BA-v2-konform ist.

## Nächster sinnvoller Schritt
- Ergaenze als naechsten kleinen BA-v2-Datenschritt gezielt belastbare Datenunterstuetzung fuer `obstacle_ground` oder `obstacle_overhang`, bevor der Slice als vollstaendiger BA-v2-Trainingskandidat behandelt wird.
