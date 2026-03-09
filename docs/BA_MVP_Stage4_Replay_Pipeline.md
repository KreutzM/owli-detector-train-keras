# BA MVP Stage-4 Replay Pipeline

## Purpose
- Make the checked-in Stage-4 `COCO replay` step directly reviewable in one place.
- Keep the Stage-4 data-prep step reviewable even after the first real Lite2 comparison run exists.
- Link the exact configs, manifests, artifacts, and the trained comparison result.

## Current Scope
- Keep the existing Stage-3 multi-source dataset:
  - `Obstacle4`
  - balanced `Mapillary Vistas`
  - `OD / Obstacle-Dataset`
- Add a small `COCO replay` subset only for the six BA-v1 rehearsal classes:
  - `person`
  - `bicycle`
  - `motorcycle`
  - `car`
  - `bus`
  - `truck`
- Do not start a new training run in this step.

Trained comparison result:
- [BA_MVP_Stage4_Baseline.md](./BA_MVP_Stage4_Baseline.md)

## Checked-In Stage-4 Files
- Replay config:
  - [`configs/coco_replay_ba_mvp_stage4.yaml`](../configs/coco_replay_ba_mvp_stage4.yaml)
- Stage-4 merge manifest:
  - [`configs/merge_ba_mvp_stage4_with_coco_replay.yaml`](../configs/merge_ba_mvp_stage4_with_coco_replay.yaml)
- Next training config:
  - [`configs/efficientdet_lite2_ba_mvp_stage4.yaml`](../configs/efficientdet_lite2_ba_mvp_stage4.yaml)
- Replay label map:
  - [`configs/label_maps/coco_replay_to_ba.yaml`](../configs/label_maps/coco_replay_to_ba.yaml)

## Verified Replay Rule
- Source COCO:
  - `work/datasets/coco2017/instances_train2017.clean.json`
- Source images:
  - `data/coco2017/images/train2017`
- Keep only:
  - `person`, `bicycle`, `motorcycle`, `car`, `bus`, `truck`
- Fixed replay heuristic:
  - drop boxes with `min_bbox_min_side < 16`
  - cap to `250` positive images per retained class

This is intentionally narrow. The goal is rehearsal against forgetting, not a return to broad COCO training.

## Verified Artifacts
- Replay subset:
  - `work/datasets/coco_replay_ba_v1_stage4/instances_ba_v1.coco.json`
  - `work/datasets/coco_replay_ba_v1_stage4/class_names.json`
  - `work/datasets/coco_replay_ba_v1_stage4/qc_report.json`
- Stage-4 merged dataset:
  - `work/datasets/ba_mvp_stage4_with_coco_replay/instances_combined.json`
  - `work/datasets/ba_mvp_stage4_with_coco_replay/instances_combined.report.json`
- Stage-4 split:
  - `work/splits/ba_mvp_stage4_with_coco_replay/splits.json`
  - `work/splits/ba_mvp_stage4_with_coco_replay/instances_train.json`
  - `work/splits/ba_mvp_stage4_with_coco_replay/instances_val.json`
  - `work/splits/ba_mvp_stage4_with_coco_replay/instances_test.json`
- Materialized training input:
  - `work/datasets/ba_mvp_stage4_with_coco_replay/instances_materialized.json`
  - `work/datasets/ba_mvp_stage4_with_coco_replay/images`
- Model Maker CSV:
  - `work/datasets/ba_mvp_stage4_with_coco_replay/modelmaker.csv`
  - `work/datasets/ba_mvp_stage4_with_coco_replay/modelmaker.class_names.json`

## Verified Counts
- Replay subset:
  - `785` images
  - `11646` annotations
  - `6` categories
- Stage-4 merged dataset:
  - `4851` images
  - `50038` annotations
  - `10` BA-v1 categories
- Stage-4 split:
  - `missing_train_classes=[]`

## What This Doc Does Not Cover
- This page stays focused on the Stage-4 replay data path.
- The trained Lite2 result is documented separately:
  - [BA_MVP_Stage4_Baseline.md](./BA_MVP_Stage4_Baseline.md)
- The current preferred multi-source baseline remains the Stage-3 run:
  - [BA_MVP_Stage3_Baseline.md](./BA_MVP_Stage3_Baseline.md)

## Current Follow-Up
- The first real Stage-4 Lite2 comparison run now exists:
  - [BA_MVP_Stage4_Baseline.md](./BA_MVP_Stage4_Baseline.md)
- Result on current repo HEAD:
  - useful negative evidence
  - current small replay subset does not outperform the verified Stage-3 baseline
