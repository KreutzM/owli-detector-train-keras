# Codex Task Report

## Ziel
- Aus den real verfuegbaren BA-v1-Datenquellen `Obstacle4`, `Mapillary Vistas` und `OD` den ersten bewusst balancierten Multi-Source-MVP-Datensatz bauen.
- Den Pfad bis direkt vor das Training repo-seitig reproduzierbar machen:
  - balancierter `Mapillary`-Baustein
  - Drei-Quellen-Merge
  - Coverage-Split
  - materialisierte Bilder
  - `ModelMaker`-CSV
- Keine neue Datenarchitektur bauen, sondern eine kleine feste Balancing-Heuristik fuer den aktuellen MVP-Stand verankern.

## Was wurde geändert?
- Neuer kleiner COCO-Balancer:
  - `src/owli_train/data/balance_coco.py`
- Neues CLI-Subcommand:
  - `python -m owli_train dataset balance-coco --config <yaml>`
  - eingebunden in `src/owli_train/cli.py`
- Neue feste Mapillary-Balance-Config:
  - `configs/balance_ba_mvp_mapillary.yaml`
- Neues Multi-Source-Merge-Manifest:
  - `configs/merge_ba_mvp_stage3_balanced_multisource.yaml`
- Neue gezielte Tests:
  - `tests/test_dataset_balance_coco.py`
- Aktualisierte Konsistenztests:
  - `tests/test_cli_help.py`
  - `tests/test_mvp_data_prep.py`
- Doku auf den real verifizierten Multi-Source-MVP-Datensatz aktualisiert:
  - `docs/MVP_Training_Plan.md`
  - `docs/runbook.md`
  - `docs/Mapillary_Vistas_Integration.md`
- `docs/reviews/Codex-Task-Report_last.md` auf diesen Task aktualisiert

## Was wurde wirklich verifiziert?
- Reale Eingangsquellen benutzt:
  - `work/datasets/obstacle4/instances_combined.json`
  - `work/datasets/mapillary_vistas_ba_v1/instances_ba_v1.coco.json`
  - `work/datasets/od_ba_v1/instances_ba_v1.coco.json`

- Vor der Implementierung real bestaetigte Schieflage:
  - `Obstacle4`
    - `1250` Bilder
    - `1912` Annotationen
  - `Mapillary`
    - `19962` Bilder
    - `629801` Annotationen
    - starke Dominanz bei `obstacle_pole` und `car`
    - sehr viele kleine Boxen
  - `OD`
    - `1592` Bilder
    - `8911` Annotationen

- Gewaehlte feste erste Balancing-Heuristik:
  - `Obstacle4` voll behalten
  - `OD` voll behalten
  - nur `Mapillary` gezielt reduzieren
  - `Mapillary`-Regel:
    - Annotationen mit `min_bbox_min_side < 16` verwerfen
    - danach pro Zielklasse hoechstens `400` positive Bilder selektieren
  - Begruendung:
    - `Mapillary` dominiert sonst `obstacle_pole` und `car`
    - `Obstacle4` bleibt die einzige verifizierte Quelle fuer `obstacle_bump`
    - `OD` ist nuetzlich, aber auf der BA-Core-Seite schmal

- Reale Balancing-Ausfuehrung:
```bash
PYTHONPATH=src python -m owli_train dataset balance-coco \
  --config configs/balance_ba_mvp_mapillary.yaml
```
  - Exit-Code: `0`
  - Erzeugte Artefakte:
    - `work/datasets/mapillary_vistas_ba_v1_mvp_balanced/instances_ba_v1.coco.json`
    - `work/datasets/mapillary_vistas_ba_v1_mvp_balanced/class_names.json`
    - `work/datasets/mapillary_vistas_ba_v1_mvp_balanced/splits.json`
    - `work/datasets/mapillary_vistas_ba_v1_mvp_balanced/qc_report.json`
  - Ergebnis:
    - Bilder: `1224`
    - Annotationen: `27597`
    - Kategorien: `9`
    - ausgewaehlte Original-Splits:
      - `train`: `1106`
      - `val`: `118`
    - ausgewaehlte Bildanzahl je Zielklasse:
      - `obstacle_fence`: `736`
      - `obstacle_hole`: `254`
      - `obstacle_pole`: `1186`
      - `bicycle`: `456`
      - `bus`: `424`
      - `car`: `1155`
      - `motorcycle`: `526`
      - `person`: `847`
      - `truck`: `400`
    - gefilterte kleine `Mapillary`-Annotationen:
      - `250432`

- Reale Validierung des balancierten `Mapillary`-Exports:
```bash
PYTHONPATH=src python -m owli_train dataset validate \
  --coco work/datasets/mapillary_vistas_ba_v1_mvp_balanced/instances_ba_v1.coco.json \
  --images-dir work/datasets/mapillary_vistas_ba_v1/images
```
  - Exit-Code: `0`
  - Ergebnis: `images=1224`, `ann=27597`, `cats=9`

- Reale Drei-Quellen-Merge-Ausfuehrung:
```bash
PYTHONPATH=src python -m owli_train dataset merge coco \
  --manifest configs/merge_ba_mvp_stage3_balanced_multisource.yaml \
  --out work/datasets/ba_mvp_stage3_balanced_multisource/instances_combined.json
```
  - Exit-Code: `0`
  - Erzeugte Artefakte:
    - `work/datasets/ba_mvp_stage3_balanced_multisource/instances_combined.json`
    - `work/datasets/ba_mvp_stage3_balanced_multisource/instances_combined.report.json`
  - Ergebnis:
    - Bilder: `4066`
    - Annotationen: `38399`
    - Kategorien: `10`
  - Source-Mix:
    - `obstacle4`: `1250` Bilder, `1912` Annotationen
    - `mapillary_vistas`: `1224` Bilder, `27578` Annotationen
    - `od_ba_v1`: `1592` Bilder, `8909` Annotationen
  - Merge-Drops:
    - `duplicate_gt_same_class=21`

- Reale Split-Ausfuehrung mit Coverage-Gate:
```bash
PYTHONPATH=src python -m owli_train dataset split \
  --coco work/datasets/ba_mvp_stage3_balanced_multisource/instances_combined.json \
  --out-dir work/splits/ba_mvp_stage3_balanced_multisource \
  --seed 1337 \
  --ensure-train-class-coverage
```
  - Exit-Code: `0`
  - Ergebnis:
    - `work/splits/ba_mvp_stage3_balanced_multisource/splits.json`
    - Split-Groessen:
      - `train=3252`
      - `val=406`
      - `test=408`
    - `missing_train_classes=[]`
    - alle `10` BA-v1-Klassen im `TRAIN` vorhanden

- Reale Materialisierung des Drei-Quellen-Datensatzes:
```bash
PYTHONPATH=src python -m owli_train dataset materialize-images \
  --coco work/datasets/ba_mvp_stage3_balanced_multisource/instances_combined.json \
  --merge-manifest configs/merge_ba_mvp_stage3_balanced_multisource.yaml \
  --out-images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --out-coco work/datasets/ba_mvp_stage3_balanced_multisource/instances_materialized.json \
  --mode auto
```
  - Exit-Code: `0`
  - Ergebnis:
    - `work/datasets/ba_mvp_stage3_balanced_multisource/images`
    - `work/datasets/ba_mvp_stage3_balanced_multisource/instances_materialized.json`
    - `images_total=4066`
    - `written=4066`
    - `symlinked=4066`
    - `copied=0`

- Reale Validierung des materialisierten Merge-Datensatzes:
```bash
PYTHONPATH=src python -m owli_train dataset validate \
  --coco work/datasets/ba_mvp_stage3_balanced_multisource/instances_materialized.json \
  --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images
```
  - Exit-Code: `0`
  - Ergebnis: `images=4066`, `ann=38399`, `cats=10`

- Reale `ModelMaker`-CSV-Ausfuehrung:
```bash
PYTHONPATH=src python -m owli_train dataset export modelmaker-csv \
  --coco work/datasets/ba_mvp_stage3_balanced_multisource/instances_materialized.json \
  --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --splits-json work/splits/ba_mvp_stage3_balanced_multisource/splits.json \
  --out work/datasets/ba_mvp_stage3_balanced_multisource/modelmaker.csv
```
  - Exit-Code: `0`
  - Erzeugte Artefakte:
    - `work/datasets/ba_mvp_stage3_balanced_multisource/modelmaker.csv`
    - `work/datasets/ba_mvp_stage3_balanced_multisource/modelmaker.class_names.json`
  - Ergebnis:
    - Zeilen: `38399`
    - Bilder: `4066`
    - Annotationen: `38399`
    - Zeilen je Set:
      - `TRAIN=30616`
      - `VAL=3909`
      - `TEST=3874`

- Reale Klassenverteilung des finalen Multi-Source-Merge-Datensatzes:
  - `obstacle_bump`: `479`
  - `obstacle_fence`: `1104`
  - `obstacle_hole`: `673`
  - `obstacle_pole`: `10878`
  - `bicycle`: `2372`
  - `bus`: `1482`
  - `car`: `9726`
  - `motorcycle`: `2594`
  - `person`: `7521`
  - `truck`: `1570`

- Reale TRAIN-Verteilung des finalen Multi-Source-Merge-Datensatzes:
  - `obstacle_bump`: `380`
  - `obstacle_fence`: `864`
  - `obstacle_hole`: `553`
  - `obstacle_pole`: `8628`
  - `bicycle`: `1853`
  - `bus`: `1149`
  - `car`: `7784`
  - `motorcycle`: `2039`
  - `person`: `6062`
  - `truck`: `1304`

- Nur statisch geprueft:
  - In diesem Task wurde kein neuer Trainingslauf gestartet.
  - `COCO replay` wurde bewusst nicht in den Multi-Source-MVP-Datensatz aufgenommen.
  - `TACO` bleibt weiter ausserhalb dieses MVP-Schritts.

## Tests
- Gezielte neue/angepasste Tests:
```bash
python -m pytest tests/test_dataset_balance_coco.py tests/test_mvp_data_prep.py tests/test_cli_help.py
```
  - Exit-Code: `0`
  - Resultat: `26 passed`

- Repo-Pflichtchecks:
```bash
python -m ruff format .
python -m ruff check .
python -m pytest
```
  - erster Lauf:
    - `python -m ruff format .`
      - Exit-Code: `0`
      - Resultat: `2 files reformatted, 60 files left unchanged`
    - `python -m ruff check .`
      - Exit-Code: `1`
      - Grund: unsortierter Importblock in `src/owli_train/cli.py`
  - direkter Fix:
```bash
python -m ruff check src/owli_train/cli.py --fix
```
    - Exit-Code: `0`
  - finaler Pflichtlauf:
    - `python -m ruff format .`
      - Exit-Code: `0`
      - Resultat: `62 files left unchanged`
    - `python -m ruff check .`
      - Exit-Code: `0`
      - Resultat: `All checks passed!`
    - `python -m pytest`
      - Exit-Code: `0`
      - Resultat: `138 passed, 5 skipped`

## Relevante Run-Kommandos
```bash
PYTHONPATH=src python -m owli_train dataset balance-coco \
  --config configs/balance_ba_mvp_mapillary.yaml
```

```bash
PYTHONPATH=src python -m owli_train dataset merge coco \
  --manifest configs/merge_ba_mvp_stage3_balanced_multisource.yaml \
  --out work/datasets/ba_mvp_stage3_balanced_multisource/instances_combined.json
```

```bash
PYTHONPATH=src python -m owli_train dataset split \
  --coco work/datasets/ba_mvp_stage3_balanced_multisource/instances_combined.json \
  --out-dir work/splits/ba_mvp_stage3_balanced_multisource \
  --seed 1337 \
  --ensure-train-class-coverage
```

```bash
PYTHONPATH=src python -m owli_train dataset materialize-images \
  --coco work/datasets/ba_mvp_stage3_balanced_multisource/instances_combined.json \
  --merge-manifest configs/merge_ba_mvp_stage3_balanced_multisource.yaml \
  --out-images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --out-coco work/datasets/ba_mvp_stage3_balanced_multisource/instances_materialized.json \
  --mode auto
```

```bash
PYTHONPATH=src python -m owli_train dataset export modelmaker-csv \
  --coco work/datasets/ba_mvp_stage3_balanced_multisource/instances_materialized.json \
  --images-dir work/datasets/ba_mvp_stage3_balanced_multisource/images \
  --splits-json work/splits/ba_mvp_stage3_balanced_multisource/splits.json \
  --out work/datasets/ba_mvp_stage3_balanced_multisource/modelmaker.csv
```

## Offene Risiken
- Der Datensatz ist deutlich kontrollierter als der volle `Mapillary`-Merge, aber `obstacle_pole` und `car` bleiben weiterhin dominante Klassen.
- `obstacle_bump` kommt weiterhin nur aus `Obstacle4`.
- `obstacle_hole` bleibt ueber alle drei Quellen hinweg schwach.
- `COCO replay` ist in diesem ersten Multi-Source-MVP-Datensatz bewusst noch nicht enthalten; schwache Rehearsal-Klassen koennen im spaeteren Training weiter unter Druck geraten.
- Die lokale `Mapillary`-Lizenz bleibt `CC BY-NC-SA` laut lokalem Lizenzfile.

## Genau ein nächster sinnvoller Schritt
- Auf Basis dieses materialisierten Multi-Source-MVP-Datensatzes jetzt den ersten echten `EfficientDet-Lite2`-Trainingslauf ohne `COCO replay` als klare Stage-3-Baseline starten und danach erst entscheiden, wie gross ein spaeterer Replay-Baustein sein darf.
