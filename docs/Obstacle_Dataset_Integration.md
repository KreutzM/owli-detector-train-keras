# Obstacle-Dataset Integration Prep

## Status on Current Repo HEAD
- No second obstacle dataset is materialized in this repo beyond `data/raw/obstacle4`.
- A repeated local search across the repo, the surrounding WSL user workspace, and common Windows user roots found only the existing Obstacle4 artifacts.
- This file therefore documents a prepared integration path, not a completed DS2 ingestion.
- As of this local machine state, there is no verified DS2 raw path, taxonomy file, or license/readme artifact to map yet.

## Why this exists
- BA-v1 is now explicit in `configs/label_contracts/ba_v1.yaml`.
- The next product-relevant data step is to strengthen the four BA core obstacle classes with another obstacle-focused dataset.
- The repo already supports the needed mechanics:
  - YOLO import via `dataset import yolo`
  - COCO normalize/validate
  - deterministic merge manifests
  - split repair with `--ensure-train-class-coverage`
  - Model Maker CSV export for the Lite2 path

## Checked-in prep artifact
- Mapping prep file: `configs/label_maps/obstacle_dataset_to_ba.yaml`
- Current intent of that file:
  - freeze the allowed BA-v1 target classes for this dataset
  - keep the future mapping constrained to BA core classes only
  - avoid an implicit taxonomy change when DS2 arrives

The checked-in `map` is intentionally empty today because the source taxonomy is not yet verified locally.

## Mapping rule for DS2
Use the Obstacle-Dataset only to strengthen BA core classes unless a later review proves otherwise.

Allowed BA-v1 targets:
- `obstacle_bump`
- `obstacle_fence`
- `obstacle_hole`
- `obstacle_pole`

Working rule:
- map only source classes that cleanly and defensibly correspond to one of the four BA core classes
- keep uncertain classes out
- do not use this dataset to backfill `person`, `bicycle`, `motorcycle`, `bus`, `car`, or `truck` unless a later PR explicitly justifies that change

## Expected local inputs before the next real integration
The next PR needs these facts verified from the local raw dataset, not guessed:
- source path
- license / redistribution constraints
- source format (`YOLO`, `COCO`, or other)
- source category vocabulary
- image root layout
- whether the dataset contains classes that map cleanly into the four BA core classes

## Next real integration path
After the local dataset is present and reviewed, the smallest real DS2 path should be:

### If the source is YOLO
```bash
python -m owli_train dataset import yolo \
  --yolo-dir <verified_obstacle_dataset_root> \
  --out work/datasets/obstacle_dataset/instances_raw.json

python -m owli_train dataset normalize \
  --coco work/datasets/obstacle_dataset/instances_raw.json \
  --images-dir <verified_obstacle_dataset_root_or_images_dir> \
  --label-map configs/label_maps/obstacle_dataset_to_ba.yaml \
  --out work/datasets/obstacle_dataset/instances_ba_v1.json

python -m owli_train dataset validate \
  --coco work/datasets/obstacle_dataset/instances_ba_v1.json \
  --images-dir <verified_obstacle_dataset_root_or_images_dir>
```

### If the source is already COCO
```bash
python -m owli_train dataset validate \
  --coco <verified_obstacle_dataset_coco.json> \
  --images-dir <verified_obstacle_dataset_images_dir>

python -m owli_train dataset normalize \
  --coco <verified_obstacle_dataset_coco.json> \
  --images-dir <verified_obstacle_dataset_images_dir> \
  --label-map configs/label_maps/obstacle_dataset_to_ba.yaml \
  --out work/datasets/obstacle_dataset/instances_ba_v1.json
```

## Merge target for the BA-v1 Lite2 path
Once `instances_ba_v1.json` exists and validates, the next merge PR should combine it with the current corrected Obstacle4 path, then continue with the existing downstream steps:
- merge COCO
- split from the merged COCO with `--ensure-train-class-coverage`
- export Model Maker CSV
- train EfficientDet-Lite2
- inspect / eval / golden detect

## What is still intentionally open
- exact DS2 source format
- exact DS2 class vocabulary
- whether DS2 needs a new import adapter or can use the current YOLO/COCO path directly
- exact merge weighting and balance relative to Obstacle4
