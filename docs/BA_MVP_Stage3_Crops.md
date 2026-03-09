# BA MVP Stage-3 Small-Object Crops

## Scope
- `Stage-3` remains the current preferred multi-source baseline.
- This path does not replace the full-image Stage-3 dataset.
- It adds a small, reproducible crop dataset for a later `Stage-3` vs. `Stage-3-plus-crops` comparison run.

## Target Classes
Current BA-v1 names only:
- `obstacle_bump`
- `obstacle_fence`
- `obstacle_hole`
- `obstacle_pole`

Older wording such as `obstacle_fence_rail` and `obstacle_hole_dropoff` stays mapped to
the current contract names `obstacle_fence` and `obstacle_hole`.

## Input and Heuristic
- Config:
  - [`configs/crop_ba_mvp_stage3_small_obstacles.yaml`](../configs/crop_ba_mvp_stage3_small_obstacles.yaml)
- Input COCO:
  - `work/splits/ba_mvp_stage3_balanced_multisource/instances_train.json`
- Input images:
  - `work/datasets/ba_mvp_stage3_balanced_multisource/images`
- Allowed source prefixes, in priority order:
  - `obstacle4`
  - `mapillary_vistas`
  - `od_ba_v1`

Crop-worthy seed rule:
- keep only the four BA-core targets above
- require a small target box by at least one fixed rule:
  - `min(width, height) <= 48`
  - or bbox area ratio `<= 0.01`
  - or bbox short-side ratio `<= 0.05`
- cap output to:
  - `200` crops per target class
  - `1` crop per source image

Crop window rule:
- square crop centered on the seed box
- side length:
  - `max(256, 4.0 * max(width, height))`
  - clamped to `[256, 512]` and to image bounds
- retain intersecting annotations only if:
  - retained area ratio `>= 0.5`
  - retained min side `>= 4`

Why this stays small:
- no generic tiling
- fixed thresholds only
- train-split only, so derived crops stay out of later Stage-3 validation and test comparisons

## Verified Export Result
Real run on repo HEAD:
- output dir:
  - `work/datasets/ba_mvp_stage3_crops`
- crop COCO:
  - `work/datasets/ba_mvp_stage3_crops/instances_ba_v1.coco.json`
- crop images:
  - `work/datasets/ba_mvp_stage3_crops/images`
- QC:
  - `work/datasets/ba_mvp_stage3_crops/qc_report.json`
- Model Maker CSV:
  - `work/datasets/ba_mvp_stage3_crops/modelmaker.csv`

Observed output:
- `528` crop images
- `3001` annotations
- `10` BA-v1 categories remain present through contextual carry-over

Selected crop targets:
- `obstacle_bump`: `3`
- `obstacle_fence`: `176`
- `obstacle_hole`: `149`
- `obstacle_pole`: `200`

Selected source mix:
- `obstacle4`: `41`
- `mapillary_vistas`: `487`
- `od_ba_v1`: `0`

Reading:
- the fixed small-object heuristic finds almost all useful crop seeds in `Mapillary`
  and a small `Obstacle4` tail
- `OD` is allowed, but on the current priority/cap setup it does not enter the first crop export
- the crop set stays materially smaller than the base Stage-3 train set

## Stage-3-plus-crops Prep Path
Prepared artifacts for the next Lite2 comparison run:
- merge manifest:
  - [`configs/merge_ba_mvp_stage3_plus_crops.yaml`](../configs/merge_ba_mvp_stage3_plus_crops.yaml)
- merged COCO:
  - `work/datasets/ba_mvp_stage3_plus_crops/instances_combined.json`
- materialized COCO:
  - `work/datasets/ba_mvp_stage3_plus_crops/instances_materialized.json`
- materialized images:
  - `work/datasets/ba_mvp_stage3_plus_crops/images`
- combined Model Maker CSV:
  - `work/datasets/ba_mvp_stage3_plus_crops/modelmaker.csv`
- next training config:
  - [`configs/efficientdet_lite2_ba_mvp_stage3_plus_crops.yaml`](../configs/efficientdet_lite2_ba_mvp_stage3_plus_crops.yaml)

Current verified prepared size:
- `4594` images
- `41400` annotations
- `10` categories

## Working Rule
- Keep `Stage-3` as the default baseline.
- Treat crops as an additional high-resolution signal for small BA-core objects only.
- Compare the next Lite2 run directly as:
  - `Stage-3`
  - vs. `Stage-3-plus-crops`

## First Comparison Result
- verified comparison run:
  - [BA_MVP_Stage3_Plus_Crops_Baseline.md](./BA_MVP_Stage3_Plus_Crops_Baseline.md)
- on the held-out Stage-3 `TEST` split, the first real crop run does not beat the Stage-3 baseline:
  - AP `0.1307 -> 0.1280`
  - AP50 `0.2325 -> 0.2276`
  - AP75 `0.1270 -> 0.1202`
  - AR100 `0.2170 -> 0.2142`
  - precision `0.2050 -> 0.2083`
  - recall `0.3735 -> 0.3684`

Practical reading:
- `obstacle_fence` improves the clearest
- `obstacle_pole` recall nudges upward, but FP load also rises
- `obstacle_hole` gets fewer FP, but also less recall
- `obstacle_bump` still remains too weak
- the crop branch stays a useful experiment path, not the new default baseline
