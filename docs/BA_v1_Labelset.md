# BA-v1 Labelset

## Purpose
- Freeze the first explicit product label contract for the BA detector path.
- Keep Android/TFLite integration on a fixed class order instead of an implicit dataset-derived order.
- Separate BA core classes from COCO-derived rehearsal classes so the next dataset work has a clear target.

Machine-readable source of truth for this contract:
- `configs/label_contracts/ba_v1.yaml`

Current evidence base:
- The latest fully verified Obstacle4 Lite2 reference run in [Obstacle4_E2E_Results.md](./Obstacle4_E2E_Results.md)

## Canonical Class Order
The BA-v1 order is fixed and intentional. Index order matters because TFLite labels are resolved positionally.

1. `obstacle_bump`
2. `obstacle_fence`
3. `obstacle_hole`
4. `obstacle_pole`
5. `bicycle`
6. `bus`
7. `car`
8. `motorcycle`
9. `person`
10. `truck`

## Class Roles
| Order | Class | Role | Why it is in BA-v1 |
| --- | --- | --- | --- |
| 1 | `obstacle_bump` | BA core | Direct obstacle type from the verified Obstacle4 GT path. |
| 2 | `obstacle_fence` | BA core | Structural barrier relevant to path traversal. |
| 3 | `obstacle_hole` | BA core | Ground opening / drop hazard relevant to safe navigation. |
| 4 | `obstacle_pole` | BA core | Vertical obstacle relevant to collision avoidance. |
| 5 | `bicycle` | COCO-critical rehearsal | Keep as a street-object rehearsal class even though the current baseline does not learn it yet. |
| 6 | `bus` | COCO-critical rehearsal | Keep as a large-vehicle rehearsal class; current baseline evidence is still weak. |
| 7 | `car` | COCO-critical rehearsal | Keep because it is the only rehearsal class with real signal in the current baseline. |
| 8 | `motorcycle` | COCO-critical rehearsal | Keep as a street-object rehearsal class despite weak current evidence. |
| 9 | `person` | COCO-critical rehearsal | Keep because it is safety-critical even though the current baseline does not yet learn it well. |
| 10 | `truck` | COCO-critical rehearsal | Keep as a large-vehicle rehearsal class despite weak current evidence. |

## Why this is BA-v1 and not a larger set
- The verified Obstacle4 path currently gives strong direct evidence only for four obstacle classes.
- The current baseline preserves all ten labels end-to-end, but only `car` shows usable rehearsal behavior.
- A wider class set would increase product and data obligations without evidence that the current data path can support it.
- BA-v1 therefore keeps the smallest labelset that still reflects:
  - the obstacle classes already grounded in the current BA dataset path
  - a small set of safety-relevant rehearsal classes we intentionally want to improve next

## What is deliberately not in BA-v1
- Full COCO-80 coverage.
- Fine-grained subclasses such as vehicle subtype or person subtype variants.
- Traffic-sign, traffic-light, and lane semantics.
- Indoor inventory classes with no clear BA-v1 requirement.
- New obstacle subclasses that do not yet have a verified dataset and mapping plan.

Everything outside the canonical ten-class list is out of scope for BA-v1 unless the product contract is explicitly revised.

## Current baseline evidence from the verified Obstacle4 Lite2 run
Source:
- [Obstacle4_E2E_Results.md](./Obstacle4_E2E_Results.md)

Practical reading of the latest full TFLite eval:
- BA core classes are the current center of gravity.
  - `obstacle_fence`, `obstacle_hole`, `obstacle_pole`, and `obstacle_bump` are learned to some extent.
  - False positives are still high, so this is not product-ready.
- Rehearsal classes are uneven.
  - `car` shows real signal.
  - `person`, `bicycle`, `motorcycle`, `bus`, and `truck` are preserved in the final label contract but are effectively not learned yet.

This is why BA-v1 keeps those classes as explicit rehearsal targets instead of silently removing them.

## Android / TFLite consequences
- For BA-v1 runs, `labels.txt` and `class_names.json` must preserve the BA-v1 order exactly.
- Android rendering must treat class index `i` as `class_names[i]` from the exported artifacts, with no local reordering.
- Eval and golden detect must continue to resolve labels from exported artifacts (`labels.txt`, `class_names.json`, metadata) rather than a separate hard-coded mobile list.
- If a future run intentionally changes the BA-v1 order or membership, that is a product-contract change and must be documented before Android integration picks it up.

## Next dataset integration priorities
These are product-driven priorities, not completed integrations.

The current primary multi-source MVP assembly plan is tracked in [MVP_Training_Plan.md](./MVP_Training_Plan.md).

### 1. Obstacle-Dataset
- Integration prep on current repo HEAD: [Obstacle_Dataset_Integration.md](./Obstacle_Dataset_Integration.md)
- Expected value:
  - Strengthen BA core classes first.
  - Reduce dependence on a single obstacle source.
- BA-v1 classes most likely to benefit:
  - `obstacle_bump`
  - `obstacle_fence`
  - `obstacle_hole`
  - `obstacle_pole`
- Likely repo work:
  - Verify license and local acquisition path.
  - Add import adapter or normalization mapping into the BA-v1 obstacle classes.
  - Review split behavior and merge compatibility with the current Obstacle4 path.
- Risks / unknowns:
  - Exact class vocabulary, layout, and label quality are not yet verified in this repo.
  - Mapping may be lossy if the source taxonomy does not align cleanly to the four BA core classes.

### 2. TACO
- Expected value:
  - Add a second non-Obstacle4 source that may improve clutter robustness and hard negatives around ground-level objects.
  - Potentially strengthen parts of the BA core path if some classes can be mapped without forcing taxonomy inflation.
- BA-v1 classes most likely to benefit:
  - Assumption pending source review: mostly `obstacle_bump` / `obstacle_hole` adjacent clutter scenarios, plus general detector robustness.
- Likely repo work:
  - Verify whether a conservative BA-v1 mapping is justified.
  - Add a label map only for categories that can be defended as BA-v1-relevant.
  - Keep non-mappable categories out rather than inflating the contract.
- Risks / unknowns:
  - This repo does not yet verify that TACO categories align cleanly to BA-v1.
  - A weak mapping could add noise faster than it adds signal.

### 3. Later optional rehearsal source
- Expected value:
  - Specifically improve the weak rehearsal classes `person`, `bicycle`, `motorcycle`, `bus`, `truck`.
- BA-v1 classes most likely to benefit:
  - the five weak rehearsal classes above, plus `car`
- Likely repo work:
  - Add a narrow label map for only the BA-v1 rehearsal classes.
  - Keep split coverage and label-order gates enabled.
- Risks / unknowns:
  - Source choice is still open in the repo; no specific dataset is frozen here yet.
  - Rehearsal-only data can distort the class balance if it is merged without care.

## Working rule for the next PRs
- Do not widen BA-v1 casually.
- Prefer improving evidence for the current ten classes over adding new labels.
- If a dataset cannot be mapped cleanly into BA-v1, document the mismatch instead of stretching the taxonomy.
