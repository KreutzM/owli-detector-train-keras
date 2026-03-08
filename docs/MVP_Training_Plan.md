# BA MVP Training Plan

## Purpose
- Define the primary near-term training path for the first BA MVP detector run.
- Keep the repo aligned on one model family, one label contract, and one multi-source data plan.
- Make the next data-integration steps explicit without pretending that all raw downloads are already verified locally.

## Fixed Product Contract
- Primary model path: `EfficientDet-Lite2` via Model Maker
- Primary export target: Android-compatible `TFLite`
- Primary label contract: [`configs/label_contracts/ba_v1.yaml`](../configs/label_contracts/ba_v1.yaml)
- Labelset rationale: [BA_v1_Labelset.md](./BA_v1_Labelset.md)
- Current verified baseline: [Obstacle4_E2E_Results.md](./Obstacle4_E2E_Results.md)
- Current verified multi-source baseline: [BA_MVP_Stage3_Baseline.md](./BA_MVP_Stage3_Baseline.md)

## Primary MVP Sources
The MVP path is no longer Obstacle4-only. Obstacle4 remains the verified reference baseline, but the next training run is intended to be multi-source.

| Source | MVP role | Current repo status | Intended BA-v1 contribution |
| --- | --- | --- | --- |
| `Obstacle4` | baseline anchor | fully verified on current repo HEAD | BA core classes + current pseudo-label bridge into rehearsal classes |
| `Mapillary Vistas` | BA supplemental source | full local `v1.2` BA-filtered export verified; `Map2/v2.0` support exists; balanced MVP subset and Stage-3 multi-source merge hook verified | strengthen BA core classes and selected rehearsal classes from street-scene data without letting Mapillary dominate the first MVP run |
| `TACO` | BA supplemental source | download / local source review pending | add BA-relevant clutter / hard-negative coverage where mappings are defensible |
| `Obstacle-Dataset / OD` | BA supplemental source | local split-VOC raw source reviewed; BA-filtered COCO export and `Obstacle4` merge hook verified | strengthen `obstacle_pole` and selected exact-match rehearsal classes from an obstacle-focused source |
| `COCO replay` | rehearsal-only replay | local `train2017.clean` replay subset and Stage-4 merge hook verified | preserve signal for `person`, `bicycle`, `motorcycle`, `car`, `bus`, `truck` without reverting to COCO-80 training |

Current concrete MVP merge hooks:
- pairwise reference hooks:
  - [`configs/merge_ba_mvp_stage2_obstacle4_mapillary.yaml`](../configs/merge_ba_mvp_stage2_obstacle4_mapillary.yaml)
  - [`configs/merge_ba_mvp_stage2_obstacle4_od.yaml`](../configs/merge_ba_mvp_stage2_obstacle4_od.yaml)
- first balanced multi-source hook:
  - [`configs/balance_ba_mvp_mapillary.yaml`](../configs/balance_ba_mvp_mapillary.yaml)
  - [`configs/merge_ba_mvp_stage3_balanced_multisource.yaml`](../configs/merge_ba_mvp_stage3_balanced_multisource.yaml)
- verified balanced source target:
  - `work/datasets/mapillary_vistas_ba_v1_mvp_balanced`
- verified balanced multi-source output:
  - `work/datasets/ba_mvp_stage3_balanced_multisource/instances_combined.json`

## Planned Training Stages

### Stage 1. Obstacle4 + pseudo
Status:
- already verified on repo HEAD

Goal:
- keep the current reproducible EfficientDet-Lite2 reference path runnable
- preserve the full 10-class BA-v1 contract through export

Expected outcome:
- stable technical baseline
- BA core classes learned to some extent
- weak rehearsal-class performance remains visible rather than hidden

### Stage 2. Add BA supplemental datasets
Sources:
- `Mapillary Vistas`
- `TACO`
- `Obstacle-Dataset / OD`

Goal:
- improve BA core data diversity without widening BA-v1
- reduce dependence on a single obstacle source

Expected outcome:
- better BA core recall / precision tradeoff than the Obstacle4-only baseline
- fewer obvious false positives on obstacle classes
- unchanged BA-v1 label order and membership

Working rule:
- until each source taxonomy is reviewed locally, keep these sources constrained to the four BA core targets
- do not use these sources to backfill rehearsal classes by default

Current exception:
- `OD` now has a reviewed local taxonomy and a partial BA-v1 mapping.
- The current OD mapping remains conservative on obstacle classes, but exact-match rehearsal labels
  (`person`, `bicycle`, `bus`, `car`, `motorbike`, `truck`) are allowed because they exist directly
  in the local XML taxonomy and do not widen BA-v1.

Current verified balancing rule for the first multi-source MVP dataset:
- keep `Obstacle4` fully
- keep `OD` fully
- keep `Mapillary` only through the checked-in balanced subset from
  [`configs/balance_ba_mvp_mapillary.yaml`](../configs/balance_ba_mvp_mapillary.yaml)
- the current heuristic is intentionally small and fixed:
  - filter Mapillary boxes with `min_bbox_min_side < 16`
  - then cap Mapillary to `400` positive images per retained target class
- rationale:
  - `Mapillary` otherwise overwhelms `obstacle_pole` and `car`
  - `Obstacle4` remains the only verified source of `obstacle_bump`
  - `OD` is useful but much smaller and narrower on the BA-core side

Current verified Stage-3 training result on repo HEAD:
- config:
  - [`configs/efficientdet_lite2_ba_mvp_stage3.yaml`](../configs/efficientdet_lite2_ba_mvp_stage3.yaml)
- run:
  - `work/runs/20260308-183140-ba-mvp-stage3-20260308`
- primary eval:
  - held-out `TEST` split from `work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json`
- primary result:
  - AP `0.1307`
  - AP50 `0.2325`
  - AP75 `0.1270`
  - all `10` BA-v1 classes produce non-zero true positives on the held-out Stage-3 test split
- comparison result on full `Obstacle4` reference set:
  - AP improves from `0.0952` to `0.2443`
  - rehearsal classes no longer stay at zero
  - false-positive pressure remains high

### Stage 3. Add small COCO replay
Goal:
- protect the BA-v1 rehearsal classes from disappearing or remaining completely untrained in the combined run

Replay classes:
- `person`
- `bicycle`
- `motorcycle`
- `car`
- `bus`
- `truck`

Expected outcome:
- maintain explicit signal for the six rehearsal classes
- avoid broad COCO-80 obligations
- keep the combined dataset focused on BA-v1 instead of turning back into a generic detector run

Current verified Stage-4 data path on repo HEAD:
- replay config:
  - [`configs/coco_replay_ba_mvp_stage4.yaml`](../configs/coco_replay_ba_mvp_stage4.yaml)
- replay output:
  - `work/datasets/coco_replay_ba_v1_stage4`
  - `785` images
  - `11646` annotations
  - `6` categories
- replay heuristic:
  - start from `work/datasets/coco2017/instances_train2017.clean.json`
  - keep only `person`, `bicycle`, `motorcycle`, `car`, `bus`, `truck`
  - filter replay boxes with `min_bbox_min_side < 16`
  - cap replay to `250` positive images per retained class
- verified Stage-4 merge hook:
  - [`configs/merge_ba_mvp_stage4_with_coco_replay.yaml`](../configs/merge_ba_mvp_stage4_with_coco_replay.yaml)
- verified Stage-4 training input prep:
  - merged COCO: `work/datasets/ba_mvp_stage4_with_coco_replay/instances_combined.json`
  - materialized COCO: `work/datasets/ba_mvp_stage4_with_coco_replay/instances_materialized.json`
  - split file: `work/splits/ba_mvp_stage4_with_coco_replay/splits.json`
  - Model Maker CSV: `work/datasets/ba_mvp_stage4_with_coco_replay/modelmaker.csv`
  - next training config:
    - [`configs/efficientdet_lite2_ba_mvp_stage4.yaml`](../configs/efficientdet_lite2_ba_mvp_stage4.yaml)

## Repo Prep Artifacts for This MVP Path
- [`configs/label_maps/obstacle4_to_ba.yaml`](../configs/label_maps/obstacle4_to_ba.yaml)
- [`configs/label_maps/mapillary_vistas_to_ba.yaml`](../configs/label_maps/mapillary_vistas_to_ba.yaml)
- [`configs/label_maps/taco_to_ba.yaml`](../configs/label_maps/taco_to_ba.yaml)
- [`configs/label_maps/obstacle_dataset_to_ba.yaml`](../configs/label_maps/obstacle_dataset_to_ba.yaml)
- [`configs/label_maps/coco_replay_to_ba.yaml`](../configs/label_maps/coco_replay_to_ba.yaml)
- [`configs/coco_replay_ba_mvp_stage4.yaml`](../configs/coco_replay_ba_mvp_stage4.yaml)
- [`configs/merge_ba_mvp_stage4_with_coco_replay.yaml`](../configs/merge_ba_mvp_stage4_with_coco_replay.yaml)
- [Obstacle_Dataset_Integration.md](./Obstacle_Dataset_Integration.md)
- [Mapillary_Vistas_Integration.md](./Mapillary_Vistas_Integration.md)

Interpretation:
- `Obstacle4` and `COCO replay` already have a concrete class-level role in the repo.
- `Mapillary Vistas` now has a concrete export path and a checked-in merge hook with `Obstacle4`.
- `Mapillary Vistas` now also has a checked-in balanced-subset config for the first multi-source MVP run.
- `TACO` is still prep-only.
- `OD` now has a real local import path, a checked-in partial mapping, and a verified merge hook with `Obstacle4`.
- `COCO replay` now has a real small subset config and a verified Stage-4 merge hook.

## Minimal Execution Plan After Downloads Finish
For each new BA supplemental source:
1. Verify the local raw path, license/readme notes, source format, and source taxonomy.
2. Fill only defensible entries in the corresponding `configs/label_maps/*.yaml`.
3. Normalize the source into `work/datasets/<source>/instances_ba_v1.json`.
4. Validate the normalized COCO and confirm only BA-v1 targets remain.
5. Merge the normalized sources into the BA-v1 MVP training dataset.
6. Split from the merged COCO with `--ensure-train-class-coverage`.
7. Export Model Maker CSV and train EfficientDet-Lite2.

For COCO replay:
1. Keep only the six rehearsal classes.
2. Build a small replay subset instead of using COCO-80 broadly.
3. Cap replay images deliberately so COCO stays smaller than the BA-focused Stage-3 sources.
4. Merge replay data only after the BA supplemental sources are normalized into BA-v1.

## What This Plan Does Not Assume Yet
- exact raw paths for `TACO`
- exact source class names for those datasets
- exact merge ratios between the new sources
- future tuning of the replay subset size for COCO

Those facts stay open until the local downloads finish and each source is reviewed from real files.

Current exception:
- `Mapillary Vistas` now has a real local source review, a full BA-filtered `v1.2` export, and a verified merge hook with `Obstacle4`.
- `Mapillary Vistas` now also has a real balanced subset export for MVP assembly:
  - `work/datasets/mapillary_vistas_ba_v1_mvp_balanced`
  - `1224` images
  - `27597` annotations
- `OD` now also has a real local source review, a verified BA-filtered export under `work/datasets/od_ba_v1`, and a verified merge hook with `Obstacle4`.
- `COCO replay` now also has a real small rehearsal subset under `work/datasets/coco_replay_ba_v1_stage4`:
  - `785` images
  - `11646` annotations
  - rehearsal-only classes: `person`, `bicycle`, `motorcycle`, `car`, `bus`, `truck`
- `TACO` remains prep-only on current repo HEAD.

## Current Risks
- The Obstacle4-only baseline is technically stable but not strong enough yet for product use.
- The new Stage-3 multi-source baseline is materially better than the Obstacle4-only run, but still not product-ready because false positives remain high.
- BA core improvements can still be undermined by weak or lossy mappings from new sources.
- COCO replay can distort the class balance if it is not kept intentionally small.
- The next combined run should be treated as an MVP integration milestone, not as a final product claim.

## Current Working Rule
- Treat `Obstacle4` as the verified reference baseline.
- Treat `Mapillary Vistas` and `OD` as reviewed BA supplements with conservative mappings on current repo HEAD.
- Use the checked-in balanced Mapillary subset for the first multi-source MVP run instead of the full raw Mapillary export.
- Treat [`BA_MVP_Stage3_Baseline.md`](./BA_MVP_Stage3_Baseline.md) as the current verified multi-source reference before adding `COCO replay`.
- Treat `TACO` as prep-only until its local taxonomy is verified.
- Treat `COCO replay` as a narrow rehearsal mechanism, not as a return to general COCO training.
- Use the checked-in Stage-4 replay subset and merge hook for the next direct Stage-3 vs. Stage-4 Lite2 comparison run.
