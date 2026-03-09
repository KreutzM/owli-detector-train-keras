# BA MVP Stage-3 Product Gate

## Scope
- Freeze the current preferred `Stage-3` detector as the current product-near candidate artifact.
- Choose one documented working threshold for product-near local checks.
- Add a small deterministic acceptance suite for repeated product-near checks.

This does not change:
- the BA-v1 contract
- Android inference code
- the current training baseline

## Candidate Artifact
- Training baseline:
  - [BA_MVP_Stage3_Baseline.md](./BA_MVP_Stage3_Baseline.md)
- Run dir:
  - `work/runs/20260308-183140-ba-mvp-stage3-20260308`
- TFLite:
  - `work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite`
- Label order:
  - unchanged BA-v1 canonical order

## Threshold Calibration
Real TFLite evals were run on the same held-out Stage-3 `TEST` split (`408` images) at:
- `0.10`
- `0.15`
- `0.20`
- `0.25`
- `0.30`

Observed operating tradeoff:

| Threshold | Precision | Recall | TP | FP | FN | FP/100 images |
| --- | --- | --- | --- | --- | --- | --- |
| `0.10` | `0.2050` | `0.3735` | `1447` | `5612` | `2427` | `1375.49` |
| `0.15` | `0.3188` | `0.3456` | `1339` | `2861` | `2535` | `701.23` |
| `0.20` | `0.4596` | `0.3036` | `1176` | `1383` | `2698` | `338.97` |
| `0.25` | `0.5672` | `0.2788` | `1080` | `824` | `2794` | `201.96` |
| `0.30` | `0.6829` | `0.2530` | `980` | `455` | `2894` | `111.52` |

Recommended current working threshold for product-near local checks:
- `0.15`

Why `0.15`:
- it cuts the aggregate FP load by about half relative to `0.10`
- it improves aggregate precision from `0.2050` to `0.3188`
- it keeps noticeably more recall than `0.20+`
- it is the least-bad compromise once the hard acceptance-suite misses are also considered
- it keeps the runtime rule simple: one global threshold, no per-class threshold scheme

Why not `0.20+` as the working point yet:
- the hardest BA-core cases fall away too quickly in the acceptance suite
- `0.20` and `0.25` become cleaner on BA-core false positives, but they miss too many of the tiny obstacle-focused checks

Why not `0.25` or `0.30`:
- `obstacle_bump`, `obstacle_fence`, `obstacle_hole`, and `obstacle_pole` all lose too much recall for the current obstacle-focused MVP path

## Provisional FP Budget
For the current Stage-3 product gate, use the held-out Stage-3 `TEST` split only as a lightweight regression guardrail:
- working threshold: `0.15`
- provisional regression budget:
  - FP/100 images at `0.15` should stay at or below about `750`

This is not a ship KPI.
It is only a small reproducible regression check for the current repo state.

## Acceptance Suite
- Suite config:
  - [`configs/acceptance/ba_mvp_stage3_rc01_suite.json`](../configs/acceptance/ba_mvp_stage3_rc01_suite.json)
- Images root:
  - `work/datasets/ba_mvp_stage3_balanced_multisource/images`
- Multi-threshold report:
  - `work/runs/20260308-183140-ba-mvp-stage3-20260308/reports/acceptance_stage3_rc01_multi_threshold.json`
- Suite size:
  - `13` images
- Composition:
  - `8` BA-core positives
  - `5` hard negatives for BA-core classes using rehearsal-only images

Intent:
- spot-check `obstacle_bump`, `obstacle_fence`, `obstacle_hole`, `obstacle_pole`
- watch for stray BA-core detections on `person`, `bicycle`, `bus`, `car`, `motorcycle`

## Real Acceptance Checks
Real TFLite detection was run on the `13`-image suite with the Stage-3 TFLite artifact at:
- `0.10`
- `0.15`
- `0.20`
- `0.25`

Observed summary:

| Threshold | BA-core focus hits | BA-core positives total | hard negatives clean | hard negatives total |
| --- | --- | --- | --- | --- |
| `0.10` | `4` | `8` | `2` | `5` |
| `0.15` | `3` | `8` | `3` | `5` |
| `0.20` | `2` | `8` | `4` | `5` |
| `0.25` | `2` | `8` | `5` | `5` |

Reading:
- no tested global threshold passes both sides well enough for a real product release
- `0.10` keeps more hard BA-core hits, but contaminates too many rehearsal-only hard negatives with BA-core false positives
- `0.20+` cleans up the negatives better, but loses too many of the hard BA-core positives
- `0.15` is the most honest middle ground for local product-near checks, not a release-ready setting
- even at `0.15`, only `3 / 8` BA-core positives hit their focus class and only `3 / 5` hard negatives stay BA-core-clean
- the remaining weak point is still recall on the hardest BA-core cases, especially `obstacle_bump`, `obstacle_fence`, and `obstacle_hole`

## Working Rule
- Keep `Stage-3` as the current preferred training baseline.
- Use threshold `0.15` as the current working operating point for local product-near TFLite checks.
- Do not treat the current Stage-3 artifact as release-ready yet.
- Re-run the Stage-3 `TEST` eval and the small acceptance suite before replacing this candidate.
