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

## Primary MVP Sources
The MVP path is no longer Obstacle4-only. Obstacle4 remains the verified reference baseline, but the next training run is intended to be multi-source.

| Source | MVP role | Current repo status | Intended BA-v1 contribution |
| --- | --- | --- | --- |
| `Obstacle4` | baseline anchor | fully verified on current repo HEAD | BA core classes + current pseudo-label bridge into rehearsal classes |
| `Mapillary Vistas` | BA supplemental source | local source verified; BA-filtered COCO converter now exists | strengthen BA core classes and selected rehearsal classes from street-scene data |
| `TACO` | BA supplemental source | download / local source review pending | add BA-relevant clutter / hard-negative coverage where mappings are defensible |
| `Obstacle-Dataset / OD` | BA supplemental source | repo prep exists, no verified local raw source on this machine | strengthen BA core classes with a second obstacle-focused source |
| `COCO replay` | rehearsal-only replay | local COCO tree already exists; replay subset not assembled yet | preserve signal for `person`, `bicycle`, `motorcycle`, `car`, `bus`, `truck` without reverting to COCO-80 training |

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

## Repo Prep Artifacts for This MVP Path
- [`configs/label_maps/obstacle4_to_ba.yaml`](../configs/label_maps/obstacle4_to_ba.yaml)
- [`configs/label_maps/mapillary_vistas_to_ba.yaml`](../configs/label_maps/mapillary_vistas_to_ba.yaml)
- [`configs/label_maps/taco_to_ba.yaml`](../configs/label_maps/taco_to_ba.yaml)
- [`configs/label_maps/obstacle_dataset_to_ba.yaml`](../configs/label_maps/obstacle_dataset_to_ba.yaml)
- [`configs/label_maps/coco_replay_to_ba.yaml`](../configs/label_maps/coco_replay_to_ba.yaml)
- [Obstacle_Dataset_Integration.md](./Obstacle_Dataset_Integration.md)
- [Mapillary_Vistas_Integration.md](./Mapillary_Vistas_Integration.md)

Interpretation:
- `Obstacle4` and `COCO replay` already have a concrete class-level role in the repo.
- `Mapillary Vistas`, `TACO`, and `OD` are checked in as conservative BA-v1 prep points, not as completed integrations.

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
3. Merge replay data only after the BA supplemental sources are normalized into BA-v1.

## What This Plan Does Not Assume Yet
- exact raw paths for `Mapillary Vistas`, `TACO`, or `OD`
- exact source class names for those datasets
- exact merge ratios between the new sources
- exact replay subset size for COCO

Those facts stay open until the local downloads finish and each source is reviewed from real files.

Current exception:
- `Mapillary Vistas` now has a real local source review and a narrow BA-filtered conversion path.
- `TACO` and `OD` still remain prep-only on current repo HEAD.

## Current Risks
- The Obstacle4-only baseline is technically stable but not strong enough yet for product use.
- BA core improvements can still be undermined by weak or lossy mappings from new sources.
- COCO replay can distort the class balance if it is not kept intentionally small.
- The next combined run should be treated as an MVP integration milestone, not as a final product claim.

## Current Working Rule
- Treat `Obstacle4` as the verified reference baseline.
- Treat `Mapillary Vistas`, `TACO`, and `OD` as BA core supplements until their local taxonomies are verified.
- Treat `COCO replay` as a narrow rehearsal mechanism, not as a return to general COCO training.
