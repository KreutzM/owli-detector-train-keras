# BA-v2 Hazard Mapping Strategy

## Purpose
- Record the first honest source-to-ontology reading for the new BA-v2 hazard target.
- Distinguish clearly between clean support, partial support, and open gaps.
- Prepare the next data/mapping reset without pretending the remap is already done.

Reference contract:
- `configs/label_contracts/ba_v2_hazard.yaml`

Important boundary:
- This document is a planning and scoping artifact.
- It is not a claim that the listed mappings are already implemented everywhere in the repo.
- The current BA-v2 MVP contract now excludes `obstacle_overhang`.
- References to `obstacle_overhang` below are historical source-fit notes, not current MVP mapping obligations.

Current checked-in first mapping slice:
- `configs/label_maps/mapillary_vistas_to_ba_v2_hazard.yaml`
- `configs/label_maps/obstacle_dataset_to_ba_v2_hazard.yaml`
- `configs/label_maps/coco_replay_to_ba_v2_hazard.yaml`
- These files deliberately cover only the cleanest currently defendable BA-v2 hazard mappings.

## Transitional Reading from BA-v1 to BA-v2 Hazard
This is a product interpretation aid, not a source-taxonomy fact.

| BA-v1 label | BA-v2 hazard reading | Confidence | Why |
| --- | --- | --- | --- |
| `obstacle_bump` | partial toward `obstacle_ground` | partial only | Some walk-surface obstacles fit, but BA-v2 ground hazards are broader than bumps. |
| `obstacle_fence` | partial toward `obstacle_barrier` | partial only | Useful barrier evidence, but BA-v2 barrier should not collapse to fences only. |
| `obstacle_hole` | partial toward `obstacle_hole_dropoff` | partial only | Safety meaning overlaps, but BA-v2 explicitly wants the drop-off framing. |
| `obstacle_pole` | `obstacle_pole` | relatively clean | Semantics stay aligned across BA-v1 and BA-v2. |
## Source Fit Summary

| Source | Clean fit | Partial fit | Weak / unclear fit | Main likely BA-v2 value |
| --- | --- | --- | --- | --- |
| `Obstacle4` | `obstacle_pole` | `obstacle_ground`, `obstacle_barrier`, `obstacle_hole_dropoff` | historical `obstacle_overhang` gap | bootstrap evidence for ground/barrier/hole/pole, but still tied to old four-class framing |
| `Mapillary Vistas` | `obstacle_barrier`, `obstacle_pole`, rehearsal classes | `obstacle_hole_dropoff` | `obstacle_ground`; historical `obstacle_overhang` gap | strongest current street-scene source for barriers, poles, and rehearsal continuity |
| `OD / Obstacle-Dataset` | `obstacle_pole`, rehearsal exact matches | none clearly beyond `obstacle_pole` | `obstacle_ground`, `obstacle_barrier`, `obstacle_hole_dropoff`; historical `obstacle_overhang` gap | narrow hazard-core support for poles plus rehearsal reinforcement |
| `COCO replay` | rehearsal classes only | none | all hazard-core classes | keeps the six rehearsal classes alive without widening the product contract |

## Source-by-Source Reading

### 1. Obstacle4
Current repo reality:
- fully verified historical baseline source
- exact source taxonomy is still the old four obstacle classes:
  - `bump`
  - `fence`
  - `hole`
  - `pole`

BA-v2 hazard fit:
- clean:
  - `pole -> obstacle_pole`
- partial:
  - `bump -> obstacle_ground`
  - `fence -> obstacle_barrier`
  - `hole -> obstacle_hole_dropoff`
Reading:
- `Obstacle4` is still useful as a bootstrap source.
- It should no longer define the product ontology.
- It is best treated as a partial hazard-core source, not the semantic anchor for BA-v2.

### 2. Mapillary Vistas
Current repo reality:
- verified local import path exists
- current checked-in map is intentionally narrow
- mapped source classes today include:
  - `construction--barrier--fence`
  - `object--support--pole`
  - `object--support--utility-pole`
  - `object--pothole`
  - the six rehearsal classes
- source classes such as `object--manhole` and rider variants are explicitly not mapped today

BA-v2 hazard fit:
- cleanest current fit:
  - barrier-like structures -> `obstacle_barrier`
  - support poles -> `obstacle_pole`
  - exact-match rehearsal classes -> rehearsal
- partial:
  - `object--pothole -> obstacle_hole_dropoff`
- weak / unclear:
  - `obstacle_ground`

Reading:
- `Mapillary` is the best current repo source for moving from a fence-only obstacle notion toward a broader barrier class.
- It also remains the strongest current street-scene source for `person` and vehicles.
- It is not enough on its own to claim robust `obstacle_ground` support.
- The first checked-in BA-v2 map therefore includes:
  - `obstacle_barrier`
  - `obstacle_hole_dropoff`
  - `obstacle_pole`
  - the six rehearsal classes

### 3. OD / Obstacle-Dataset
Current repo reality:
- verified local import path exists
- checked-in mapping is conservative
- clean hazard-core mapping today is effectively:
  - `pole`
  - `warning_column`
- exact-match rehearsal labels are also mapped:
  - `person`
  - `bicycle`
  - `bus`
  - `car`
  - `motorbike`
  - `truck`
- many source classes are intentionally left unmapped

BA-v2 hazard fit:
- clean:
  - `pole` / `warning_column -> obstacle_pole`
  - exact-match rehearsal classes -> rehearsal
- unclear or currently unsuitable:
  - `obstacle_ground`
  - `obstacle_barrier`
  - `obstacle_hole_dropoff`

Reading:
- `OD` remains useful, but mostly for `obstacle_pole` and rehearsal continuity.
- It is not currently a good primary source for the broader BA-v2 hazard-core reset.
- The first checked-in BA-v2 map therefore keeps OD narrow:
  - `obstacle_pole`
  - the six rehearsal classes

### 4. COCO replay
Current repo reality:
- verified small replay pipeline exists
- checked-in map is already constrained to:
  - `person`
  - `bicycle`
  - `motorcycle`
  - `car`
  - `bus`
  - `truck`

BA-v2 hazard fit:
- clean:
  - rehearsal classes only
- not suitable:
  - any hazard-core class

Reading:
- `COCO replay` remains valid under BA-v2 hazard for the same narrow purpose as before.
- It should not be stretched into hazard-core evidence.
- The first checked-in BA-v2 map keeps replay identical in scope:
  - rehearsal only

## Practical Reset Implications
- `Obstacle4` should be demoted from ontology anchor to partial bootstrap source.
- `Mapillary` looks strongest for `obstacle_barrier`, `obstacle_pole`, and rehearsal continuity.
- `OD` looks strongest for `obstacle_pole` plus exact-match rehearsal support.
- `COCO replay` stays rehearsal-only.
- The current BA-v2 MVP can be supported without `obstacle_overhang`, because that class has now been intentionally deferred out of scope.

## What This Document Does Not Freeze
- no promise that every BA-v2 hazard class already has enough data
- no promise that a one-step remap from BA-v1 artifacts will be semantically clean
- no claim that the next best run config is already known

## Next Step Boundary
- The next real step is the first BA-v2 MVP training run on the current four-hazard-core candidate.
- Do not silently reintroduce deferred classes such as `obstacle_overhang` into the MVP without a new product and data decision.
