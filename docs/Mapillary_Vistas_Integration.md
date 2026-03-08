# Mapillary Vistas Integration

## Status on Current Repo HEAD
- Local source path verified: `data/DataSets/Map`
- Additional local source path verified: `data/DataSets/Map2`
- Local edition identified from bundled files: `Mapillary Vistas Research edition v1.2`
- Additional local edition identified from bundled files: `Mapillary Vistas Research edition v2.0`
- Local license file says: `CC BY-NC-SA`
- Current repo support: BA-filtered conversion into COCO detection for the existing EfficientDet / Model Maker path

This is an initial narrow detector import, not a full generic Mapillary ingestion layer.

## Chosen source representation
The converter uses:
- `training/panoptic/panoptic_2018.json`
- `validation/panoptic/panoptic_2018.json`
- `training/images/*.jpg`
- `validation/images/*.jpg`

Why this path:
- `panoptic_2018.json` already provides per-segment `bbox`, `category_id`, `area`, and image linkage.
- That is smaller and more reliable for the current detector use case than reconstructing boxes from raw segmentation masks.
- `testing/` is ignored because the local dataset does not ship labels for it.

For the `Map2` layout the converter now also supports:
- `training/v1.2/panoptic/panoptic_2018.json`
- `validation/v1.2/panoptic/panoptic_2018.json`
- `training/v2.0/panoptic/panoptic_2020.json`
- `validation/v2.0/panoptic/panoptic_2020.json`

Working rule:
- default behavior stays conservative and prefers `v1.2` when both `v1.2` and `v2.0` are present
- use `--annotation-version v2.0` explicitly when you want the newer taxonomy

## Current mapped source classes
Configured in [`configs/label_maps/mapillary_vistas_to_ba.yaml`](../configs/label_maps/mapillary_vistas_to_ba.yaml):

| Mapillary source class | BA target |
| --- | --- |
| `construction--barrier--fence` | `obstacle_fence` |
| `object--support--pole` | `obstacle_pole` |
| `object--support--utility-pole` | `obstacle_pole` |
| `object--pothole` | `obstacle_hole` |
| `human--person` | `person` |
| `object--vehicle--bicycle` | `bicycle` |
| `object--vehicle--bus` | `bus` |
| `object--vehicle--car` | `car` |
| `object--vehicle--motorcycle` | `motorcycle` |
| `object--vehicle--truck` | `truck` |

## Intentionally not mapped in this first pass
- `object--manhole`
- rider classes such as `human--rider--bicyclist` and `human--rider--motorcyclist`
- every other Mapillary category outside the narrow whitelist above

Reason:
- keep the first converter aligned to the current BA-v1 contract
- avoid speculative mappings
- keep the source focused on BA-relevant obstacle and rehearsal classes

## Export behavior
- source splits used: `training`, `validation`
- `testing` ignored
- images are exported with long side capped at `1600 px`
- aspect ratio is preserved
- bounding boxes are scaled with the same resize factor
- images without remaining mapped annotations are dropped
- invalid / empty boxes are dropped

## CLI
Canonical MVP export target:
- `work/datasets/mapillary_vistas_ba_v1`

```bash
python -m owli_train dataset import mapillary-vistas \
  --mapillary-dir data/DataSets/Map \
  --out-dir work/datasets/mapillary_vistas_ba_v1 \
  --label-map configs/label_maps/mapillary_vistas_to_ba.yaml \
  --max-long-side 1600
```

Explicit `Map2/v2.0` run:
```bash
python -m owli_train dataset import mapillary-vistas \
  --mapillary-dir data/DataSets/Map2 \
  --out-dir data/processed/mapillary_ba_v2_0 \
  --label-map configs/label_maps/mapillary_vistas_to_ba.yaml \
  --annotation-version v2.0 \
  --max-long-side 1600
```

Useful bounded verification run:
```bash
python -m owli_train dataset import mapillary-vistas \
  --mapillary-dir data/DataSets/Map \
  --out-dir data/processed/mapillary_ba_v1_sample \
  --label-map configs/label_maps/mapillary_vistas_to_ba.yaml \
  --max-long-side 1600 \
  --limit-images-per-split 100
```

## Output artifacts
- `instances_ba_v1.coco.json`
- `annotations_train.coco.json`
- `annotations_val.coco.json`
- `splits.json`
- `class_names.json`
- `qc_report.json`
- resized images under `images/training/` and `images/validation/`

## MVP merge hook
The current repo hook for the next combined BA dataset is:
- [`configs/merge_ba_mvp_stage2_obstacle4_mapillary.yaml`](../configs/merge_ba_mvp_stage2_obstacle4_mapillary.yaml)

This manifest treats:
- `Obstacle4` as the 10-class anchor source
- `Mapillary Vistas` as the resized BA-filtered supplemental source

Recommended merge step after export:
```bash
python -m owli_train dataset merge coco \
  --manifest configs/merge_ba_mvp_stage2_obstacle4_mapillary.yaml \
  --out work/datasets/ba_mvp_stage2_obstacle4_mapillary/instances_combined.json
```

## Real local sample verification
Bounded run executed on this machine:
- output dir: `data/processed/mapillary_ba_v1_sample`
- limit: `100` images per split
- result:
  - combined images: `200`
  - combined annotations: `6378`
  - categories: `9`

QC highlights from `qc_report.json`:
- `training`
  - images: `100`
  - annotations: `3228`
  - strongest classes in this sample: `obstacle_pole`, `car`, `person`
- `validation`
  - images: `100`
  - annotations: `3150`
  - `obstacle_hole` appears, but only once in this sample slice

This verifies that the converter writes resized images, scaled boxes, and a BA-filtered COCO intermediate on the real local Mapillary source.

Additional bounded `Map2/v2.0` run executed on this machine:
- output dir: `data/processed/mapillary_ba_v2_0_sample`
- limit: `100` images per split
- `annotation_version`: `v2.0`
- result:
  - combined images: `200`
  - combined annotations: `6535`
  - categories: `9`

Important v2.0 note:
- `v2.0` is not a drop-in duplicate of the old label names.
- Example: person is represented as `human--person--individual` instead of `human--person`.
- The repo converter now handles that explicitly, but only when the source is routed to the `v2.0` annotation tree.

## Real local full export verification
Full `Map/v1.2` export executed on this machine:
- output dir: `work/datasets/mapillary_vistas_ba_v1`
- `annotation_version`: `v1.2`
- combined images: `19962`
- combined annotations: `629801`
- categories: `9`
- exported image tree size: about `7.65 GB`

Split-level results from `qc_report.json`:
- `training`
  - source images scanned: `18000`
  - exported images: `17966`
  - dropped after BA filter: `34`
  - annotations: `567060`
- `validation`
  - source images scanned: `2000`
  - exported images: `1996`
  - dropped after BA filter: `4`
  - annotations: `62741`

Class-distribution highlights from the exported COCO:
- all expected Mapillary target classes occur
- strongest classes:
  - `obstacle_pole`: `383513`
  - `car`: `148636`
  - `person`: `58686`
- weakest retained class:
  - `obstacle_hole`: `479`

Quality notes:
- image loss after filtering is very small: `38 / 20000` images
- the dataset is strongly skewed toward `obstacle_pole` and `car`
- many exported boxes are small:
  - `min_side < 8`: `207230`
  - `min_side < 16`: `327376`
  - `area < 32^2`: `340540`
- `obstacle_bump` is still absent in Mapillary and must come from another BA-v1 source such as `Obstacle4`

## Real local merge verification
The checked-in Stage-2 manifest was exercised on this machine:
- manifest: [`configs/merge_ba_mvp_stage2_obstacle4_mapillary.yaml`](../configs/merge_ba_mvp_stage2_obstacle4_mapillary.yaml)
- merged output: `work/datasets/ba_mvp_stage2_obstacle4_mapillary/instances_combined.json`
- merge report: `work/datasets/ba_mvp_stage2_obstacle4_mapillary/instances_combined.report.json`
- merged result:
  - images: `21212`
  - annotations: `631297`
  - categories: `10`

Merge interpretation:
- `Mapillary` contributes the large 9-class BA-filtered supplemental block
- `Obstacle4` restores the missing BA-v1 class `obstacle_bump`
- the combined split written via `--ensure-train-class-coverage` has no missing classes in `TRAIN`
