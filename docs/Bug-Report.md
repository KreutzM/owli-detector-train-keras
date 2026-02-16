# Bug Report: EfficientDet-Lite2 (Model Maker) fine-tune collapses on COCO val2017

Date: 2026-02-16  
Branch: `TEST7GPU-TRAIN-EVAL`  
Commit: `34479afde4a8257ca4f3bee16b4bdf009658fce3`  

## Summary

An end-to-end EfficientDet-Lite2 fine-tune (TensorFlow Lite Model Maker) completes successfully and exports a TFLite model, but the resulting model performs *significantly worse* on COCO `val2017` than both:

- yesterday’s fine-tuned model, and
- the official/pretrained EfficientDet-Lite2 baseline model.

The failure mode is not “slightly worse metrics”, but a **class-collapse / missing-major-classes** pattern (e.g. `person`, `car`, `bicycle`, `bus`, `truck` effectively absent), which strongly suggests a **class-index / label-map alignment issue** somewhere in the training/export/inference stack.

This report is meant to give a reviewer enough context to pinpoint the root cause in the next step.

## Impact / Severity

- Blocks using EfficientDet-Lite2 as a fine-tuning base (results are not usable).
- Risk of silently shipping a model with an incorrect Android contract (class indices/labels mismatch).

## Environment

- Host: `Linux Michael 6.6.87.2-microsoft-standard-WSL2 x86_64`
- GPU (nvidia-smi): `NVIDIA GeForce RTX 3060`, driver `576.80`
- Model Maker venv: `.venv-modelmaker-py39` (TensorFlow `2.8.4`, GPU visible)
- Training run used the GPU Docker wrapper (artifacts are root-owned in the run dir).

## Runs / Artifacts

### “Bad” run (today)

- Run ID: `20260216-084911`
- Run dir: `work/runs/20260216-084911`
- Model: `work/runs/20260216-084911/artifacts/model.tflite`
- Labels: `work/runs/20260216-084911/artifacts/labels.txt`
- Eval report (val2017, 5000 imgs): `work/runs/20260216-084911/reports/eval_efficientdet_tflite_val5000.json`

### Comparison: “Good” run (yesterday)

- Run dir: `work/runs/20260215-005450-coco2017-train`
- Model: `work/runs/20260215-005450-coco2017-train/artifacts/model.tflite`
- Labels: `work/runs/20260215-005450-coco2017-train/artifacts/labels.txt`
- Eval (val2017 compare) was recorded here:
  - `work/reports/val2017_compare/fine_tuned_eval.json`

### Baseline model

- Baseline: `work/models/efficientdet_lite2_baseline.tflite`
- Eval (val2017 compare): `work/reports/val2017_compare/baseline_eval.json`

## Reproduction

### Evaluate the exported TFLite on val2017

Assumes COCO layout:

- `data/coco2017/val2017/*.jpg`
- `data/coco2017/annotations/instances_val2017.json`

Run (Model Maker env / docker wrapper):

```bash
sudo HOME=/home/michael bash scripts/modelmaker_gpu_docker.sh run -- \
  eval efficientdet-tflite \
  data/coco2017/annotations/instances_val2017.json \
  data/coco2017/val2017 \
  work/runs/20260216-084911/artifacts/model.tflite \
  --limit-images 5000 \
  --score-threshold 0.3 \
  --noise-thresholds 0.05,0.1,0.3 \
  --max-detections-per-image 100 \
  --num-threads 12 \
  --out work/runs/20260216-084911/reports/eval_efficientdet_tflite_val5000.json
```

### Inspect TFLite contract

```bash
sudo HOME=/home/michael bash scripts/modelmaker_gpu_docker.sh run -- \
  inspect tflite work/runs/20260216-084911/artifacts/model.tflite
```

Observed for the bad run:

- `builtin_ops_only: true`
- input preprocessing in eval report: `input_size=448`, `dtype=uint8`, `letterbox_square`, `normalization=none`

### Raw class-index probe (to confirm the issue is in the model outputs)

Run in the Model Maker venv (no training, just inference on a small subset):

```bash
source .venv-modelmaker-py39/bin/activate
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from collections import Counter
import tensorflow as tf
from owli_train.tflite_detect import create_tflite_runtime, run_tflite_detection

model = Path("work/runs/20260216-084911/artifacts/model.tflite")
labels = [l.strip() for l in Path("work/runs/20260216-084911/artifacts/labels.txt").read_text().splitlines() if l.strip()]
imgs = sorted(Path("data/coco2017/val2017").glob("*.jpg"))[:120]

rt = create_tflite_runtime(model_path=model, tf=tf, num_threads=12)
counts = Counter()
for img in imgs:
    dets, _ = run_tflite_detection(runtime=rt, image_path=img, score_threshold=0.0, max_detections=100)
    for d in dets:
        counts[d.class_index] += 1

print("unique_class_indices", len(counts))
print("min_idx", min(counts), "max_idx", max(counts))
print("top10", [(i, labels[i] if 0 <= i < len(labels) else f'class_{i}', c) for i, c in counts.most_common(10)])
PY
```

Observed in this environment:

- `unique_class_indices 47`, `min_idx 1`, `max_idx 77`
- **class index 0 never appeared** in the probe window (120 images, threshold 0.0)

## Expected vs Actual

### Expected

- Fine-tuned model should be at least “baseline-ish” and must not completely lose `person`/`car`/etc on COCO `val2017`.
- Class index semantics should match `labels.txt` (Android contract + eval mapping).

### Actual (bad run)

From `work/runs/20260216-084911/reports/eval_efficientdet_tflite_val5000.json`:

- COCO `val2017` (5000 images):
  - `AP=0.0444`, `AP50=0.0753`, `AR100=0.0758`
- Noise metric (FP/100 images):
  - @0.05: `1998.98`
  - @0.10: `1303.82`
  - @0.30: `358.46`
- Aggregate @`score_threshold=0.3`:
  - `TP=1378`, `FP=17923`, `FN=35403`
  - Precision `0.0714`, Recall `0.0375`

#### Comparison (val2017, 5000 images)

From `work/reports/val2017_compare/*`:

| Model | AP | AP50 | AR100 | FP/100 @0.3 | Precision @0.3 | Recall @0.3 |
| --- | ---:| ---:| ---:| ---:| ---:| ---:|
| Today fine-tuned (`20260216-084911`) | 0.0444 | 0.0753 | 0.0758 | 358.46 | 0.0714 | 0.0375 |
| Yesterday fine-tuned (`20260215-005450-coco2017-train`) | 0.0884 | 0.1516 | 0.1398 | 153.28 | 0.5882 | 0.2976 |
| Baseline (`efficientdet_lite2_baseline.tflite`) | 0.0528 | 0.0807 | 0.0644 | 296.22 | 0.3787 | 0.2455 |

## Key Observations / Evidence

### 1) “Missing major classes” pattern

In the bad run’s per-class summary (val2017, 5000 images), many of the dominant COCO categories have:

- `TP=0`, `FP=0`, high `FN`

Examples (from the computed per-class totals at `score_threshold=0.3`):

- `person`: `TP=0`, `FP=0`, `FN=11004`
- `car`: `TP=0`, `FP=0`, `FN=1932`
- `bicycle`: `TP=0`, `FP=0`, `FN=316`

This is hard to explain as “just needs more steps” and looks like a semantics/mapping issue or a training target corruption.

### 2) Raw output probe shows class index 0 absent

For the bad run, on a 120-image probe at `score_threshold=0.0`, `class_index=0` never appears.

For the baseline model (`work/models/efficientdet_lite2_baseline.tflite`), the same style probe on 40 images did show index `0` present.

This suggests the problem exists before the COCO evaluation logic (i.e., it’s not just pycocotools/mapping).

### 3) Model Maker `label_map` differs between the “good” and “bad” runs

From Model Maker training configs:

- “Good” run: `work/runs/20260215-005450-coco2017-train/logs/modelmaker/config.yaml`
  - `label_map` starts at `1: hot dog, 2: dog, ...` (non-COCO order)
- “Bad” run: `work/runs/20260216-084911/logs/modelmaker/config.yaml`
  - `label_map` starts at `1: person, 2: bicycle, ...` (COCO order)

Important: both runs’ *input* label-map snapshot files show a COCO-ordered `class_names` list:

- `work/runs/20260215-005450-coco2017-train/label_map_input_snapshot.json`
- `work/runs/20260216-084911/label_map_input_snapshot.json`

So the difference is in the Model Maker dataloader’s resolved `label_map` / label-id assignment (not in the declared target taxonomy).

### 4) Training signal suggests generalization failure / target mismatch

Bad run end-of-training line (from the console output):

- `loss ≈ 0.9609`
- `val_loss ≈ 2.3857` (val cls loss dominates)

Yesterday’s run (recorded in `docs/COCO2017_Eval_Results.md`) had:

- `val_loss ≈ 0.8508`

## Leading Hypotheses (ranked)

1) **Label-ID / class-index semantics mismatch across stages** (Model Maker training ↔ exported TFLite ↔ our inference/eval):
   - Model Maker assigns label IDs starting at 1 (background=0 is reserved), and may or may not shift/normalize them at export.
   - Our inference/eval path assumes `class_index` is a 0-based index into `labels.txt`.
   - The observed “no index 0 ever” in raw outputs and the “dead person class” symptom are consistent with a semantic mismatch.

2) **Training run differed materially from the prior run due to label-map ordering**:
   - The “good” run’s resolved label map is not COCO-ordered, yet val2017 eval looked much better.
   - The “bad” run used a COCO-ordered label map, yet val2017 eval collapses.
   - Reviewer should identify *why* the resolved label map differs (code path difference, CSV ordering, caching behavior, or Model Maker behavior).

3) **We may be writing `labels.txt` that does not match the actual exported model’s class order**:
   - Today’s run writes `labels.txt` in COCO order.
   - If export preserves a different order (or different numeric ids), eval + Android contract become inconsistent.

## What the Reviewer Should Do Next

Suggested, concrete checks to pinpoint the root cause:

1) **Confirm the TFLite `detection_classes` semantics** for both models (baseline vs bad run):
   - Print the raw `detection_classes` output tensor values for 1 image and compare to expected label IDs.
   - Determine whether outputs are:
     - 0-based contiguous class indices (0..79),
     - 1-based label IDs (1..80),
     - COCO category IDs (1..90 with gaps),
     - or COCO category IDs minus 1.

2) **Validate `labels.txt` correctness vs embedded metadata**:
   - Check if the exported TFLite embeds labels/metadata that imply a different ordering than `artifacts/labels.txt`.
   - If yes, our contract and eval mapping must use the embedded ordering (or we must fix export/writing logic).

3) **Explain why the “good” run had a non-COCO-ordered `label_map`** despite a COCO-ordered `label_map_input_snapshot.json`:
   - The good run lacks `train_canonicalized.csv` in `work/runs/20260215-005450-coco2017-train/artifacts`, while the bad run has it.
   - This points to a code-path or version difference between runs; identify the exact change.

4) **Run a minimal A/B experiment** (small, reviewer-only):
   - Train for a tiny number of steps (e.g. 100–200) with:
     - the “old” behavior (no CSV class-order canonicalization / label order derived from CSV scan), and
     - the “new” behavior (canonicalization to COCO order),
   - then compare raw class-index probes + a small eval (limit-images 128).

## Secondary Issue (not the root cause, but impacts tooling)

Some files under `work/runs/20260216-084911` are root-owned due to running the Docker wrapper via `sudo`.

Symptoms:

- `eval_default_map_128` failed to write its JSON report with:
  - `PermissionError: ... work/runs/20260216-084911/reports/eval_default_map_128.json`

Fix is operational (ownership/umask), but should be kept separate from the model-quality bug.

