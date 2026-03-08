# Mapillary Vistas Integration

## Status on Current Repo HEAD
- Local source path verified: `data/DataSets/Map`
- Local edition identified from bundled files: `Mapillary Vistas Research edition v1.2`
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
```bash
python -m owli_train dataset import mapillary-vistas \
  --mapillary-dir data/DataSets/Map \
  --out-dir data/processed/mapillary_ba_v1 \
  --label-map configs/label_maps/mapillary_vistas_to_ba.yaml \
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
