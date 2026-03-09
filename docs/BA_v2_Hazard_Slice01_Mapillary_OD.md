# BA-v2 Hazard Slice 01: Mapillary + OD

## Purpose
- Record the first real BA-v2 hazard data slice on current repo HEAD.
- Start the BA-v2 product path with the smallest defensible real sources.
- Prepare the first partial BA-v2 training input without starting a training run yet.

## Source Selection
Included in this slice:
- `Mapillary Vistas`
- `OD / Obstacle-Dataset`

Explicitly not included in this slice:
- `Obstacle4`
  - kept as historical baseline and possible later bootstrap source
  - not needed for the first clean BA-v2 slice
- `COCO replay`
  - rehearsal-only source
  - intentionally not part of this first BA-v2 data slice

## Why These Sources
- `Mapillary` is the strongest currently verified source for:
  - `obstacle_barrier`
  - `obstacle_pole`
  - cautious `obstacle_hole_dropoff`
  - the six rehearsal classes
- `OD` is the strongest currently verified source for:
  - `obstacle_pole`
  - exact-match rehearsal classes
- `Obstacle4` remains useful historically, but it still carries the old four-class framing and is therefore not the cleanest first BA-v2 entry point.

## Mapping Used

### Transitional remap from verified BA-v1 source exports
Because the repo already contains verified BA-v1 source exports for `Mapillary` and `OD`, this first real BA-v2 slice converts those verified exports into BA-v2 in contract order.

Mapping file:
- `configs/label_maps/ba_v1_non_obstacle4_export_to_ba_v2_hazard.yaml`

Used mappings:
- `obstacle_fence -> obstacle_barrier`
- `obstacle_hole -> obstacle_hole_dropoff`
- `obstacle_pole -> obstacle_pole`
- `person -> person`
- `bicycle -> bicycle`
- `motorcycle -> motorcycle`
- `car -> car`
- `bus -> bus`
- `truck -> truck`

Important boundary:
- `obstacle_bump` is intentionally not mapped here.
- This transitional remap is only for BA-v1 exports that do not contain `obstacle_bump`.
- That is why this slice uses `Mapillary + OD`, not `Obstacle4`.

### Explicitly not mapped
From `Mapillary`:
- `object--manhole`
- rider classes
- person-group
- all other uncertain classes outside the checked-in BA-v2 slice

From `OD`:
- `spherical_roadblock`
- `reflective_cone`
- `ashcan`
- `fire_hydrant`
- `tricycle`
- `stop_sign`
- `dog`

Also intentionally not added from `OD`:
- no barrier mapping yet
- no ground mapping yet
- no overhang mapping yet
- no hole/dropoff mapping yet

## Real Commands
Executed on current repo HEAD:

```bash
PYTHONPATH=src python -m owli_train dataset normalize \
  --coco work/datasets/mapillary_vistas_ba_v1/instances_ba_v1.coco.json \
  --images-dir work/datasets/mapillary_vistas_ba_v1/images \
  --label-map configs/label_maps/ba_v1_non_obstacle4_export_to_ba_v2_hazard.yaml \
  --contract configs/label_contracts/ba_v2_hazard.yaml \
  --out work/datasets/mapillary_vistas_ba_v2_hazard_source/instances_normalized.json

PYTHONPATH=src python -m owli_train dataset normalize \
  --coco work/datasets/od_ba_v1/instances_ba_v1.coco.json \
  --images-dir work/datasets/od_ba_v1/images \
  --label-map configs/label_maps/ba_v1_non_obstacle4_export_to_ba_v2_hazard.yaml \
  --contract configs/label_contracts/ba_v2_hazard.yaml \
  --out work/datasets/od_ba_v2_hazard_source/instances_normalized.json

PYTHONPATH=src python -m owli_train dataset balance-coco \
  --config configs/balance_ba_v2_hazard_mapillary_slice01.yaml

PYTHONPATH=src python -m owli_train dataset balance-coco \
  --config configs/balance_ba_v2_hazard_od_slice01.yaml

PYTHONPATH=src python -m owli_train dataset merge coco \
  --manifest configs/merge_ba_v2_hazard_slice01_mapillary_od.yaml \
  --out work/datasets/ba_v2_hazard_slice01_mapillary_od/instances_combined.json \
  --report-out work/datasets/ba_v2_hazard_slice01_mapillary_od/instances_combined.report.json

PYTHONPATH=src python -m owli_train dataset split \
  --coco work/datasets/ba_v2_hazard_slice01_mapillary_od/instances_combined.json \
  --out-dir work/splits/ba_v2_hazard_slice01_mapillary_od \
  --seed 1337 \
  --ensure-train-class-coverage \
  --write-coco

PYTHONPATH=src python -m owli_train dataset materialize-images \
  --coco work/datasets/ba_v2_hazard_slice01_mapillary_od/instances_combined.json \
  --merge-manifest configs/merge_ba_v2_hazard_slice01_mapillary_od.yaml \
  --out-images-dir work/datasets/ba_v2_hazard_slice01_mapillary_od/images \
  --out-coco work/datasets/ba_v2_hazard_slice01_mapillary_od/instances_materialized.json \
  --mode auto

PYTHONPATH=src python -m owli_train dataset validate \
  --coco work/datasets/ba_v2_hazard_slice01_mapillary_od/instances_materialized.json \
  --images-dir work/datasets/ba_v2_hazard_slice01_mapillary_od/images

PYTHONPATH=src python -m owli_train dataset export modelmaker-csv \
  --coco work/datasets/ba_v2_hazard_slice01_mapillary_od/instances_materialized.json \
  --images-dir work/datasets/ba_v2_hazard_slice01_mapillary_od/images \
  --splits-json work/splits/ba_v2_hazard_slice01_mapillary_od/splits.json \
  --out work/datasets/ba_v2_hazard_slice01_mapillary_od/modelmaker.csv
```

## Generated Artifacts
- `work/datasets/mapillary_vistas_ba_v2_hazard_source/instances_normalized.json`
- `work/datasets/od_ba_v2_hazard_source/instances_normalized.json`
- `work/datasets/mapillary_vistas_ba_v2_hazard_slice01_balanced/instances_ba_v1.coco.json`
- `work/datasets/od_ba_v2_hazard_slice01/instances_ba_v1.coco.json`
- `work/datasets/ba_v2_hazard_slice01_mapillary_od/instances_combined.json`
- `work/datasets/ba_v2_hazard_slice01_mapillary_od/instances_materialized.json`
- `work/datasets/ba_v2_hazard_slice01_mapillary_od/modelmaker.csv`
- `work/datasets/ba_v2_hazard_slice01_mapillary_od/modelmaker.class_names.json`
- `work/splits/ba_v2_hazard_slice01_mapillary_od/splits.json`

## Observed Output

### Source slices
Mapillary BA-v2 slice:
- images: `957`
- annotations: `21707`
- categories: `9`

OD BA-v2 slice:
- images: `1592`
- annotations: `8911`
- categories: `7`

### Combined BA-v2 slice
- images: `2549`
- annotations: `30604`
- categories present: `9`
- split sizes:
  - `TRAIN`: `2039`
  - `VAL`: `254`
  - `TEST`: `256`

Combined category order:
1. `obstacle_barrier`
2. `obstacle_hole_dropoff`
3. `obstacle_pole`
4. `person`
5. `bicycle`
6. `motorcycle`
7. `car`
8. `bus`
9. `truck`

Per-class annotations in the combined slice:
- `obstacle_barrier`: `571`
- `obstacle_hole_dropoff`: `254`
- `obstacle_pole`: `8611`
- `person`: `6478`
- `bicycle`: `2024`
- `motorcycle`: `2244`
- `car`: `7888`
- `bus`: `1240`
- `truck`: `1294`

Source image mix:
- `mapillary_vistas`: `957`
- `od_ba_v2_hazard`: `1592`

## What This Slice Actually Supports
Real BA-v2 hazard-core support in this first slice:
- `obstacle_barrier`
- `obstacle_hole_dropoff`
- `obstacle_pole`

Still open or absent:
- `obstacle_ground`
- `obstacle_overhang`

Reading:
- `obstacle_pole` is already strongly supported.
- `obstacle_barrier` is present, but currently dominated by `Mapillary`.
- `obstacle_hole_dropoff` is present, but still relatively small.
- Rehearsal support is already substantial without using `COCO replay`.

## Outcome
- This is a real BA-v2-compatible data slice, not just a contract doc.
- It is small enough to review and already useful as the first partial BA-v2 training input.
- It is not yet a complete BA-v2 training candidate for the full hazard contract, because `obstacle_ground` and `obstacle_overhang` are still missing.
- BA-v1 Stage-3 and Stage-4 remain historical baselines; this slice does not replace their evidence.
