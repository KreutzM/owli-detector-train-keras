# BA-v2 MVP Train Candidate

## Purpose
- Record the first train-ready BA-v2 MVP candidate on current repo HEAD.
- Freeze the current MVP reading as four hazard-core classes plus six rehearsal classes.
- Keep this file focused on the pre-run candidate state.
- Post-run baseline results now live in [BA_v2_MVP_Baseline.md](./BA_v2_MVP_Baseline.md).

## Current BA-v2 MVP Contract
Hazard-core:
- `obstacle_ground`
- `obstacle_barrier`
- `obstacle_hole_dropoff`
- `obstacle_pole`

Rehearsal:
- `person`
- `bicycle`
- `motorcycle`
- `car`
- `bus`
- `truck`

Explicitly not part of the current MVP:
- `obstacle_overhang`
  - deferred on purpose because no defendable data path exists on current repo HEAD

## Real Candidate Artifacts
- materialized COCO:
  - `work/datasets/ba_v2_mvp_candidate/instances_materialized.json`
- materialized images:
  - `work/datasets/ba_v2_mvp_candidate/images`
- Model Maker CSV:
  - `work/datasets/ba_v2_mvp_candidate/modelmaker.csv`
- Model Maker class names:
  - `work/datasets/ba_v2_mvp_candidate/modelmaker.class_names.json`
- split file used for CSV export:
  - `work/splits/ba_v2_hazard_slice02_mapillary_od_ground/splits.json`
- next training config:
  - `configs/efficientdet_lite2_ba_v2_mvp.yaml`

## Real Commands
Executed on current repo HEAD:

```bash
PYTHONPATH=src python -m owli_train dataset materialize-images \
  --coco work/datasets/ba_v2_hazard_slice02_mapillary_od_ground/instances_combined.json \
  --merge-manifest configs/merge_ba_v2_hazard_slice02_mapillary_od_obstacle4_ground_materialize.yaml \
  --out-images-dir work/datasets/ba_v2_mvp_candidate/images \
  --out-coco work/datasets/ba_v2_mvp_candidate/instances_materialized.json \
  --mode symlink

PYTHONPATH=src python -m owli_train dataset validate \
  --coco work/datasets/ba_v2_mvp_candidate/instances_materialized.json \
  --images-dir work/datasets/ba_v2_mvp_candidate/images

PYTHONPATH=src python -m owli_train dataset export modelmaker-csv \
  --coco work/datasets/ba_v2_mvp_candidate/instances_materialized.json \
  --images-dir work/datasets/ba_v2_mvp_candidate/images \
  --splits-json work/splits/ba_v2_hazard_slice02_mapillary_od_ground/splits.json \
  --out work/datasets/ba_v2_mvp_candidate/modelmaker.csv
```

## Observed Output
- images: `3799`
- annotations: `32231`
- categories present: `10`
- split sizes:
  - `TRAIN`: `3039`
  - `VAL`: `379`
  - `TEST`: `381`

Category order:
1. `obstacle_ground`
2. `obstacle_barrier`
3. `obstacle_hole_dropoff`
4. `obstacle_pole`
5. `person`
6. `bicycle`
7. `motorcycle`
8. `car`
9. `bus`
10. `truck`

Per-class annotations:
- `obstacle_ground`: `479`
- `obstacle_barrier`: `939`
- `obstacle_hole_dropoff`: `673`
- `obstacle_pole`: `8972`
- `person`: `6478`
- `bicycle`: `2024`
- `motorcycle`: `2244`
- `car`: `7888`
- `bus`: `1240`
- `truck`: `1294`

Source image mix:
- `mapillary_vistas`: `957`
- `od_ba_v2_hazard`: `1592`
- `obstacle4`: `1250`

## Readout
- All four current BA-v2 MVP hazard-core classes are now backed by real data on current repo HEAD.
- The candidate is materialized and exported to Model Maker CSV.
- This is the first real BA-v2 MVP training candidate in the repo.
- The candidate has now been consumed by the first real BA-v2 MVP Lite2 baseline run:
  - [BA_v2_MVP_Baseline.md](./BA_v2_MVP_Baseline.md)
