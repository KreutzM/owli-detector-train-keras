# Codex Task Report

## Ziel
- Einen kleinen echten `COCO replay`-Baustein fuer die sechs BA-v1-Rehearsal-Klassen bauen.
- Daraus einen reproduzierbaren Stage-4-Datenpfad fuer den naechsten `EfficientDet-Lite2`-Vergleichslauf vorbereiten.
- Den Pfad real bis `merge`, `split`, `materialize-images` und `modelmaker.csv` verifizieren, noch ohne neuen Trainingslauf.

## Was wurde geändert?
- Neuer kleiner `COCO replay`-Importer:
  - `src/owli_train/data/coco_replay.py`
- Neue CLI:
  - `python -m owli_train dataset import coco-replay --config ...`
  - eingebunden in `src/owli_train/cli.py`
- Neue Replay-Config:
  - `configs/coco_replay_ba_mvp_stage4.yaml`
- Neues Stage-4-Merge-Manifest:
  - `configs/merge_ba_mvp_stage4_with_coco_replay.yaml`
- Naechste Trainingsconfig vorbereitet:
  - `configs/efficientdet_lite2_ba_mvp_stage4.yaml`
- Neue Tests:
  - `tests/test_dataset_import_coco_replay.py`
  - `tests/test_mvp_data_prep.py` erweitert
- Doku aktualisiert:
  - `docs/MVP_Training_Plan.md`
  - `docs/runbook.md`
- `docs/reviews/Codex-Task-Report_last.md` auf diesen Task aktualisiert

## Was wurde wirklich verifiziert?
- Lokaler COCO2017-Bestand real geprueft:
```bash
find data/coco2017 -maxdepth 3 \( -type f -o -type d \) | sort | sed -n '1,120p'
python - <<'PY'
import json
from pathlib import Path
raw=Path('data/coco2017/annotations/instances_train2017.json')
clean=Path('work/datasets/coco2017/instances_train2017.clean.json')
for p in [raw, clean]:
    obj=json.loads(p.read_text())
    print(p, len(obj['images']), len(obj['annotations']), len(obj['categories']))
PY
```
  - Exit-Code: `0`
  - Ergebnis:
    - `data/coco2017/images/train2017` lokal vorhanden
    - `data/coco2017/annotations/instances_train2017.json` lokal vorhanden
    - `work/datasets/coco2017/instances_train2017.clean.json` lokal vorhanden
    - raw: `118287` Bilder / `860001` Annotationen / `80` Kategorien
    - clean: `118287` Bilder / `859999` Annotationen / `80` Kategorien

- Reale Klassenhaeufigkeits- und Heuristikpruefung fuer das Replay:
```bash
python - <<'PY'
import json
from collections import Counter,defaultdict
from pathlib import Path
p=Path('work/datasets/coco2017/instances_train2017.clean.json')
obj=json.loads(p.read_text())
keep={'person','bicycle','car','motorcycle','bus','truck'}
id_to_name={c['id']:c['name'] for c in obj['categories']}
counts=Counter(); imgs=defaultdict(set)
for ann in obj['annotations']:
    n=id_to_name[ann['category_id']]
    if n in keep:
        counts[n]+=1
        imgs[n].add(ann['image_id'])
print('ann_counts', counts)
print('img_counts', {k:len(v) for k,v in imgs.items()})
PY
```
  - Exit-Code: `0`
  - Ergebnis:
    - `person` dominiert roh deutlich
    - die sechs Replay-Klassen sind real vorhanden
    - daraus wurde die kleine feste Heuristik abgeleitet:
      - `min_bbox_min_side=16`
      - `max_positive_images_per_class=250`

- Reale Replay-Erzeugung:
```bash
PYTHONPATH=src python -m owli_train dataset import coco-replay \
  --config configs/coco_replay_ba_mvp_stage4.yaml
```
  - Exit-Code: `0`
  - erzeugte Artefakte:
    - `work/datasets/coco_replay_ba_v1_stage4/instances_ba_v1.coco.json`
    - `work/datasets/coco_replay_ba_v1_stage4/class_names.json`
    - `work/datasets/coco_replay_ba_v1_stage4/qc_report.json`
  - reales Ergebnis:
    - `785` Bilder
    - `11646` Annotationen
    - `6` Kategorien

- Reale Replay-Validierung:
```bash
PYTHONPATH=src python -m owli_train dataset validate \
  --coco work/datasets/coco_replay_ba_v1_stage4/instances_ba_v1.coco.json \
  --images-dir data/coco2017/images/train2017
```
  - Exit-Code: `0`

- Reale Replay-QC-Auswertung:
```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('work/datasets/coco_replay_ba_v1_stage4/qc_report.json')
obj=json.loads(p.read_text())
print(obj['selected_annotation_counts'])
print(obj['selected_image_counts'])
PY
```
  - Exit-Code: `0`
  - Ergebnis:
    - Annotationen:
      - `bicycle=2031`
      - `bus=1217`
      - `car=1348`
      - `motorcycle=2363`
      - `person=3648`
      - `truck=1039`
    - positive Bilder:
      - `bicycle=289`
      - `bus=272`
      - `car=346`
      - `motorcycle=277`
      - `person=570`
      - `truck=250`

- Reale Stage-4-Merge-Erzeugung:
```bash
PYTHONPATH=src python -m owli_train dataset merge coco \
  --manifest configs/merge_ba_mvp_stage4_with_coco_replay.yaml \
  --out work/datasets/ba_mvp_stage4_with_coco_replay/instances_combined.json
```
  - Exit-Code: `0`
  - erzeugte Artefakte:
    - `work/datasets/ba_mvp_stage4_with_coco_replay/instances_combined.json`
    - `work/datasets/ba_mvp_stage4_with_coco_replay/instances_combined.report.json`
  - reales Ergebnis:
    - `4851` Bilder
    - `50038` Annotationen
    - `10` Kategorien
    - Quellenmix:
      - `Obstacle4=1250` Bilder / `1912` Annotationen kept
      - `Mapillary balanced=1224` Bilder / `27578` Annotationen kept
      - `OD=1592` Bilder / `8909` Annotationen kept
      - `COCO replay=785` Bilder / `11639` Annotationen kept

- Reale Stage-4-Validierung:
```bash
PYTHONPATH=src python -m owli_train dataset validate \
  --coco work/datasets/ba_mvp_stage4_with_coco_replay/instances_combined.json
```
  - Exit-Code: `0`

- Reale Stage-4-Split-Erzeugung:
```bash
PYTHONPATH=src python -m owli_train dataset split \
  --coco work/datasets/ba_mvp_stage4_with_coco_replay/instances_combined.json \
  --out-dir work/splits/ba_mvp_stage4_with_coco_replay \
  --seed 1337 \
  --ensure-train-class-coverage \
  --write-coco
```
  - Exit-Code: `0`
  - erzeugte Artefakte:
    - `work/splits/ba_mvp_stage4_with_coco_replay/splits.json`
    - `work/splits/ba_mvp_stage4_with_coco_replay/instances_train.json`
    - `work/splits/ba_mvp_stage4_with_coco_replay/instances_val.json`
    - `work/splits/ba_mvp_stage4_with_coco_replay/instances_test.json`
  - reales Ergebnis:
    - `missing_train_classes=[]`
    - Split-Verteilung:
      - `train=3880` Bilder / `40282` Annotationen
      - `val=485` Bilder / `4578` Annotationen
      - `test=486` Bilder / `5178` Annotationen

- Reale Stage-4-Materialisierung:
```bash
rm -rf work/datasets/ba_mvp_stage4_with_coco_replay/images
PYTHONPATH=src python -m owli_train dataset materialize-images \
  --coco work/datasets/ba_mvp_stage4_with_coco_replay/instances_combined.json \
  --merge-manifest configs/merge_ba_mvp_stage4_with_coco_replay.yaml \
  --out-images-dir work/datasets/ba_mvp_stage4_with_coco_replay/images \
  --out-coco work/datasets/ba_mvp_stage4_with_coco_replay/instances_materialized.json \
  --mode auto
```
  - Exit-Code: `0`
  - Ergebnis:
    - `4851` Bilder geschrieben
    - `4851` Symlinks
    - `0` Skips

- Reale Materialized-Validierung:
```bash
PYTHONPATH=src python -m owli_train dataset validate \
  --coco work/datasets/ba_mvp_stage4_with_coco_replay/instances_materialized.json \
  --images-dir work/datasets/ba_mvp_stage4_with_coco_replay/images
```
  - Exit-Code: `0`

- Reale ModelMaker-CSV-Erzeugung:
```bash
PYTHONPATH=src python -m owli_train dataset export modelmaker-csv \
  --coco work/datasets/ba_mvp_stage4_with_coco_replay/instances_materialized.json \
  --images-dir work/datasets/ba_mvp_stage4_with_coco_replay/images \
  --splits-json work/splits/ba_mvp_stage4_with_coco_replay/splits.json \
  --out work/datasets/ba_mvp_stage4_with_coco_replay/modelmaker.csv
```
  - Exit-Code: `0`
  - erzeugte Artefakte:
    - `work/datasets/ba_mvp_stage4_with_coco_replay/modelmaker.csv`
    - `work/datasets/ba_mvp_stage4_with_coco_replay/modelmaker.class_names.json`
  - Ergebnis:
    - `50038` CSV-Zeilen

- Zielgerichtete Code-Tests:
```bash
python -m pytest tests/test_dataset_import_coco_replay.py tests/test_mvp_data_prep.py tests/test_cli_help.py
```
  - Exit-Code: `0`
  - Ergebnis: `28 passed`

## Was wurde nur statisch geprüft?
- Die neue Stage-4-Trainingsconfig `configs/efficientdet_lite2_ba_mvp_stage4.yaml` wurde noch nicht fuer einen echten Trainingslauf verwendet.
- Es wurde kein neuer `EfficientDet-Lite2`-Trainingslauf gestartet.
- Es wurde kein neues `TFLite`-Artefakt erzeugt.

## Exportierte Artefakte / Zielpfade
- Replay-Datensatz:
  - `work/datasets/coco_replay_ba_v1_stage4/instances_ba_v1.coco.json`
  - `work/datasets/coco_replay_ba_v1_stage4/class_names.json`
  - `work/datasets/coco_replay_ba_v1_stage4/qc_report.json`
- Stage-4-Merge:
  - `work/datasets/ba_mvp_stage4_with_coco_replay/instances_combined.json`
  - `work/datasets/ba_mvp_stage4_with_coco_replay/instances_combined.report.json`
- Stage-4-Splits:
  - `work/splits/ba_mvp_stage4_with_coco_replay/splits.json`
  - `work/splits/ba_mvp_stage4_with_coco_replay/instances_train.json`
  - `work/splits/ba_mvp_stage4_with_coco_replay/instances_val.json`
  - `work/splits/ba_mvp_stage4_with_coco_replay/instances_test.json`
- Stage-4-materialized:
  - `work/datasets/ba_mvp_stage4_with_coco_replay/instances_materialized.json`
  - `work/datasets/ba_mvp_stage4_with_coco_replay/images`
- Stage-4-Trainingseingang:
  - `work/datasets/ba_mvp_stage4_with_coco_replay/modelmaker.csv`
  - `configs/efficientdet_lite2_ba_mvp_stage4.yaml`

## Offene Risiken
- `person` bleibt auch im kleinen Replay die haeufigste Klasse; das Replay ist bewusst klein, aber nicht perfekt gleichverteilt.
- `COCO replay` staerkt nur die sechs Rehearsal-Klassen; `obstacle_bump`, `obstacle_fence`, `obstacle_hole` und `obstacle_pole` bekommen daraus bewusst kein neues Signal.
- Der eigentliche Qualitaetseffekt des Replay-Bausteins ist noch unbewiesen, bis ein echter Stage-4-Trainingslauf gegen denselben Eval-Rahmen wie Stage-3 gelaufen ist.

## Genau ein nächster sinnvoller Schritt
- Fuehre den naechsten echten `EfficientDet-Lite2`-Lauf mit `configs/efficientdet_lite2_ba_mvp_stage4.yaml` aus und vergleiche ihn direkt gegen die bestehende Stage-3-Baseline auf demselben gehaltenen `TEST`-Split.
