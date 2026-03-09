# BA-v2 Hazard Slice 02: Obstacle4 Ground Bootstrap

## Purpose
- Record the next real BA-v2 hazard data step after `Slice01`.
- Add the first real local data support for `obstacle_ground`.
- Keep `obstacle_overhang` explicitly open because no defensible local source was found in this task.

## Local Source Check
Checked local source evidence:
- `data/raw/obstacle4/extracted/data.yaml`
  - local raw class list is only `bump`, `fence`, `hole`, `pole`
- `data/processed/mapillary_ba_v2_0_sample/instances_ba_v1.coco.json`
  - checked-in local sample still only exposes `obstacle_fence`, `obstacle_hole`, `obstacle_pole` plus rehearsal classes
- `work/datasets/od_ba_v1/qc_report.json`
  - locally reviewed OD raw classes include `pole`, `warning_column`, `reflective_cone`, `spherical_roadblock`, `ashcan`, `fire_hydrant`, `tricycle`, `stop_sign`, `dog`

Reading:
- `obstacle_ground` can be bootstrapped locally from `Obstacle4` only.
- No checked local source in this repo currently gives a clean `obstacle_overhang` path.

## Mapping Used
Mapping file:
- `configs/label_maps/obstacle4_gt_to_ba_v2_hazard_ground_bootstrap.yaml`

Used mappings:
- `obstacle_bump -> obstacle_ground`
- `obstacle_fence -> obstacle_barrier`
- `obstacle_hole -> obstacle_hole_dropoff`
- `obstacle_pole -> obstacle_pole`

Important boundary:
- `obstacle_bump -> obstacle_ground` is treated as a narrow subset bootstrap only.
- This does not claim that `Obstacle4` now defines BA-v2.
- No mapping is claimed here for `obstacle_overhang`.

## Generated Artifacts
- `work/datasets/obstacle4_ba_v2_hazard_ground_source/instances_normalized.json`
- `work/datasets/obstacle4_ba_v2_ground_slice02/instances_ba_v1.coco.json`
- `work/datasets/obstacle4_ba_v2_ground_slice02/class_names.json`
- `work/datasets/obstacle4_ba_v2_ground_slice02/qc_report.json`
- `work/datasets/ba_v2_hazard_slice02_mapillary_od_ground/instances_combined.json`
- `work/datasets/ba_v2_hazard_slice02_mapillary_od_ground/instances_combined.report.json`
- `work/splits/ba_v2_hazard_slice02_mapillary_od_ground/splits.json`

## Observed Output

### Obstacle4 ground bootstrap source
- images: `1250`
- annotations: `1627`
- categories: `4`
- per-class annotations:
  - `obstacle_ground`: `479`
  - `obstacle_barrier`: `368`
  - `obstacle_hole_dropoff`: `419`
  - `obstacle_pole`: `361`

### Combined BA-v2 slice after adding ground bootstrap
- images: `3799`
- annotations: `32231`
- categories present: `10`
- split sizes:
  - `TRAIN`: `3039`
  - `VAL`: `379`
  - `TEST`: `381`

Combined category order:
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

Per-class annotations in the combined slice:
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

## What This Slice Actually Supports
Real BA-v2 hazard-core support after this slice:
- `obstacle_ground`
- `obstacle_barrier`
- `obstacle_hole_dropoff`
- `obstacle_pole`

Still open:
- `obstacle_overhang`

Reading:
- BA-v2 now has a real local path for `obstacle_ground`, but only through a narrow legacy bootstrap.
- `obstacle_overhang` remains fully unsupported on current repo HEAD.

## Training Readout
- This slice moves BA-v2 materially closer to a first training candidate.
- It still does not make BA-v2 a full hazard-contract training candidate, because `obstacle_overhang` remains missing.
- In this run, the combined COCO and deterministic splits were rebuilt successfully.
- The final materialize-images / ModelMaker-CSV step for this enlarged slice was not completed in this task because the image-materialization step hit repeated WSL filesystem I/O stalls on the local machine.
