# Obstacle-Dataset Integration

## Status on Current Repo HEAD
- The local OD raw source is now verified on this machine at `/mnt/e/DataSets/Obstacle Dataset`.
- A first real BA-v1 import path exists via `python -m owli_train dataset import obstacle-dataset`.
- The importer produces a BA-filtered COCO detection dataset under `work/datasets/od_ba_v1`.
- A concrete merge hook with `Obstacle4` also exists:
  - `configs/merge_ba_mvp_stage2_obstacle4_od.yaml`

This is a real integration step, not a placeholder anymore.

## What was actually found locally
The local dataset is mixed, not a single clean canonical export:
- split VOC-style XML annotations:
  - `ann-train/`
  - `ann-val/`
  - `ann-test/`
- split image folders:
  - `img-train/`
  - `img-val/`
  - `img-test/`
- additional YOLO-style text labels:
  - `label-train/`
  - `label-val/`
  - `label-test/`
- older legacy VOC-like tree:
  - `Annotations/`
  - `JPEGImages/`
  - `ImageSets/Main/`
- additional fallback image tree:
  - `OD-test/JPEGImages/`

No local dataset `README`, `LICENSE`, or equivalent redistribution note was found alongside this OD download.

## Import strategy used in the repo
The checked-in importer intentionally uses the split VOC XML path, not the YOLO text labels.

Chosen source:
- `ann-train`, `ann-val`, `ann-test`

Image resolution roots:
- split-local root first:
  - `img-train`, `img-val`, `img-test`
- then fallback roots:
  - `JPEGImages`
  - `OD-test/JPEGImages`

Why this path was chosen:
- the VOC XML files contain explicit human-readable class names
- the YOLO label files are present, but no local class-id legend was found
- the older `Annotations/JPEGImages` tree is smaller and not the richest locally available OD source

Working rule:
- keep the import narrow and defensible
- prefer a smaller real subset over guessing missing labels or class ids

## Verified local taxonomy
Classes seen in the split VOC XML files:
- `ashcan`
- `bicycle`
- `bus`
- `car`
- `dog`
- `fire_hydrant`
- `motorbike`
- `person`
- `pole`
- `reflective_cone`
- `spherical_roadblock`
- `stop_sign`
- `tricycle`
- `truck`
- `warning_column`

## Current BA-v1 mapping
Source of truth:
- `configs/label_maps/obstacle_dataset_to_ba.yaml`

Mapped today:
- `pole -> obstacle_pole`
- `warning_column -> obstacle_pole`
- `bicycle -> bicycle`
- `bus -> bus`
- `car -> car`
- `motorbike -> motorcycle`
- `person -> person`
- `truck -> truck`

Left intentionally unmapped:
- `reflective_cone`
- `spherical_roadblock`
- `ashcan`
- `fire_hydrant`
- `tricycle`
- `dog`
- `stop_sign`

Reason:
- these classes are either not clean BA-v1 matches or would stretch `obstacle_bump` / `obstacle_fence` / `obstacle_hole` too aggressively
- `OD` therefore helps `obstacle_pole` plus exact-match rehearsal classes on current repo HEAD

## Real export result on this machine
Command:
```bash
python -m owli_train dataset import obstacle-dataset \
  --dataset-dir '/mnt/e/DataSets/Obstacle Dataset' \
  --out-dir work/datasets/od_ba_v1 \
  --label-map configs/label_maps/obstacle_dataset_to_ba.yaml \
  --mode auto
```

Export artifacts:
- `work/datasets/od_ba_v1/instances_ba_v1.coco.json`
- `work/datasets/od_ba_v1/annotations_train.coco.json`
- `work/datasets/od_ba_v1/annotations_val.coco.json`
- `work/datasets/od_ba_v1/annotations_test.coco.json`
- `work/datasets/od_ba_v1/splits.json`
- `work/datasets/od_ba_v1/class_names.json`
- `work/datasets/od_ba_v1/qc_report.json`

Real validated export:
- images: `1592`
- annotations: `8911`
- categories: `7`
- categories present:
  - `obstacle_pole`
  - `bicycle`
  - `bus`
  - `car`
  - `motorcycle`
  - `person`
  - `truck`

Per split:
- `train`
  - exported images: `471`
  - exported annotations: `1789`
  - skipped XMLs because local image missing: `72`
  - dropped images after BA filter: `457`
- `val`
  - exported images: `436`
  - exported annotations: `2841`
  - skipped XMLs because local image missing: `504`
  - dropped images after BA filter: `173`
- `test`
  - exported images: `685`
  - exported annotations: `4281`
  - skipped XMLs because local image missing: `201`
  - dropped images after BA filter: `302`
  - duplicate filename conflicts resolved against the real local image size: `2`

## Real merge hook with Obstacle4
Manifest:
- `configs/merge_ba_mvp_stage2_obstacle4_od.yaml`

Command:
```bash
python -m owli_train dataset merge coco \
  --manifest configs/merge_ba_mvp_stage2_obstacle4_od.yaml \
  --out work/datasets/ba_mvp_stage2_obstacle4_od/instances_combined.json
```

Real merged result:
- images: `2842`
- annotations: `10821`
- categories: `10`
- merge report:
  - `work/datasets/ba_mvp_stage2_obstacle4_od/instances_combined.report.json`
- duplicate GT same class drops: `2`

Coverage split check:
```bash
python -m owli_train dataset split \
  --coco work/datasets/ba_mvp_stage2_obstacle4_od/instances_combined.json \
  --out-dir work/splits/ba_mvp_stage2_obstacle4_od \
  --seed 1337 \
  --ensure-train-class-coverage
```

Real result:
- `missing_train_classes: []`
- all 10 BA-v1 classes are present in `TRAIN`

## What OD currently does and does not contribute
Current real contribution:
- strengthens `obstacle_pole`
- adds additional exact-match signal for:
  - `person`
  - `bicycle`
  - `bus`
  - `car`
  - `motorcycle`
  - `truck`

Current non-contribution:
- no defensible mapping to `obstacle_bump`
- no defensible mapping to `obstacle_fence`
- no defensible mapping to `obstacle_hole`

Those classes still need other data sources.

## Current risks
- The local OD download is incomplete relative to its XML set; many XMLs reference missing images.
- No local license/readme artifact was found in this download.
- The taxonomy is partially useful, but much of the obstacle-focused content stays unmapped under the current conservative BA-v1 rule.
- `OD` should therefore be treated as a partial BA-v1 supplement, not as a full obstacle-core replacement for `Obstacle4`.
