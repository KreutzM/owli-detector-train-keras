# BA-v2 Hazard Labelset

## Purpose
- Define the preferred product ontology for the current BA MVP path.
- Move the product contract away from the old Obstacle4-shaped taxonomy.
- Keep the target labelset small, fixed, and explicit for the first real BA-v2 MVP training candidate.

Machine-readable source of truth for this contract:
- `configs/label_contracts/ba_v2_hazard.yaml`

Current status:
- preferred product target for the repo
- first real BA-v2 MVP training candidate now exists on current repo HEAD
- Stage-3 and Stage-4 remain historical BA-v1 baselines, not BA-v2 hazard evidence

## Canonical Class Order
The BA-v2 hazard order is fixed and intentional. Index order matters because TFLite labels are resolved positionally.

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

## Class Roles
| Order | Class | Role | Why it is in BA-v2 hazard |
| --- | --- | --- | --- |
| 1 | `obstacle_ground` | Hazard core | Covers walk-surface obstacles more broadly than the old `obstacle_bump` label. |
| 2 | `obstacle_barrier` | Hazard core | Covers traversal-blocking barrier structures without tying the product contract to `fence` only. |
| 3 | `obstacle_hole_dropoff` | Hazard core | Keeps ground-opening / drop hazard semantics explicit instead of flattening them to the old `obstacle_hole` wording. |
| 4 | `obstacle_pole` | Hazard core | Keeps a narrow vertical collision-hazard class with direct practical relevance. |
| 5 | `person` | Rehearsal | Safety-critical dynamic actor kept as a narrow rehearsal class. |
| 6 | `bicycle` | Rehearsal | Street-scene rehearsal class retained without widening to generic COCO coverage. |
| 7 | `motorcycle` | Rehearsal | Street-scene rehearsal class retained without widening to generic COCO coverage. |
| 8 | `car` | Rehearsal | Current strongest vehicle rehearsal signal, retained for continuity. |
| 9 | `bus` | Rehearsal | Large-vehicle rehearsal class retained as part of the small replay bundle. |
| 10 | `truck` | Rehearsal | Large-vehicle rehearsal class retained as part of the small replay bundle. |

## Separation from BA-v1
BA-v1 was a verified interim path. BA-v2 hazard is a deliberate product reset.

What changes:
- The product ontology is now hazard-centered, not Obstacle4-centered.
- The old BA-v1 obstacle classes were too tightly coupled to the available Obstacle4 labels:
  - `obstacle_bump`
  - `obstacle_fence`
  - `obstacle_hole`
- Those three names are no longer product-contract classes.
- `obstacle_pole` remains because it still reads as a real hazard class instead of a dataset-specific subtype.

Practical reading:
- `obstacle_ground` is not just a rename of `obstacle_bump`.
- `obstacle_barrier` is broader than `obstacle_fence`.
- `obstacle_hole_dropoff` is closer to the intended safety meaning than the narrower BA-v1 `obstacle_hole`.
- `obstacle_overhang` was considered during the BA-v2 reset, but is now intentionally deferred out of the current MVP because the repo still has no defendable data path for it.

## Why BA-v2 Hazard Stays Small
- The product target should stay reviewable and trainable with the current repo tooling.
- A larger ontology would create mapping obligations the current sources do not yet justify.
- Returning to COCO-80 would hide the real product problem instead of solving it.
- The rehearsal set stays narrow on purpose:
  - `person`
  - `bicycle`
  - `motorcycle`
  - `car`
  - `bus`
  - `truck`

## Android / TFLite Consequences
- For a future BA-v2 hazard run, `labels.txt` and `class_names.json` must preserve the BA-v2 hazard order exactly.
- Android must keep resolving labels from exported artifacts, not from a locally hard-coded list.
- The current verified Android/TFLite artifacts in the repo remain BA-v1 artifacts until a real BA-v2 hazard model is trained and exported.
- A switch from BA-v1 to BA-v2 hazard is therefore a product-contract change, not just a training tweak.

## Current Boundary
- This file does not claim that BA-v2 hazard is already dataset-verified.
- This file does not claim that the deferred `obstacle_overhang` class is solved.
- The current BA-v2 MVP contract is intentionally limited to four hazard-core classes plus six rehearsal classes.
- The next real repo step is the first BA-v2 MVP training run, not another ontology expansion.

See also:
- [BA_v2_Hazard_Mapping_Strategy.md](./BA_v2_Hazard_Mapping_Strategy.md)
- [BA_v1_Labelset.md](./BA_v1_Labelset.md)
- [MVP_Training_Plan.md](./MVP_Training_Plan.md)
