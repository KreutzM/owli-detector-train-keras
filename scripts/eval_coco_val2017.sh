#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python3}"
if ! command -v "$PYTHON_EXE" >/dev/null 2>&1; then
  PYTHON_EXE="python"
fi
MODELMAKER_PYTHON_EXE="${MODELMAKER_PYTHON_EXE:-$PYTHON_EXE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

COCO_ROOT="data/coco2017"
FINE_TUNED_MODEL=""
BASELINE_MODEL=""
LIMIT_IMAGES=5000
MAX_DETECTIONS=100
SCORE_THRESHOLD=0.3
NOISE_THRESHOLDS="0.05,0.1,0.3"
OUT_MD="docs/COCO2017_Val_Eval_Report.md"
OUT_JSON=""
CATEGORY_MAP=""
DOWNLOAD_COCO_IF_MISSING=0
DOWNLOAD_BASELINE_IF_MISSING=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/eval_coco_val2017.sh \
    --fine-tuned-model <path/to/fine_tuned.tflite> \
    --baseline-model <path/to/baseline.tflite> \
    [--coco-root data/coco2017] \
    [--limit-images 5000] \
    [--max-detections 100] \
    [--score-threshold 0.3] \
    [--noise-thresholds 0.05,0.1,0.3] \
    [--download-coco-if-missing] \
    [--download-baseline-if-missing] \
    [--category-map <path/to/map.yaml>] \
    [--out-md docs/COCO2017_Val_Eval_Report.md] \
    [--out-json docs/COCO2017_Val_Eval_Report.json]

Notes:
  - Baseline model is user-supplied via --baseline-model.
  - If using split environments, set MODELMAKER_PYTHON_EXE to the Model Maker interpreter.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --coco-root)
      COCO_ROOT="$2"
      shift 2
      ;;
    --fine-tuned-model)
      FINE_TUNED_MODEL="$2"
      shift 2
      ;;
    --baseline-model)
      BASELINE_MODEL="$2"
      shift 2
      ;;
    --limit-images)
      LIMIT_IMAGES="$2"
      shift 2
      ;;
    --max-detections)
      MAX_DETECTIONS="$2"
      shift 2
      ;;
    --score-threshold)
      SCORE_THRESHOLD="$2"
      shift 2
      ;;
    --noise-thresholds)
      NOISE_THRESHOLDS="$2"
      shift 2
      ;;
    --download-coco-if-missing)
      DOWNLOAD_COCO_IF_MISSING=1
      shift
      ;;
    --download-baseline-if-missing)
      DOWNLOAD_BASELINE_IF_MISSING=1
      shift
      ;;
    --category-map)
      CATEGORY_MAP="$2"
      shift 2
      ;;
    --out-md)
      OUT_MD="$2"
      shift 2
      ;;
    --out-json)
      OUT_JSON="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$FINE_TUNED_MODEL" || -z "$BASELINE_MODEL" ]]; then
  echo "[ERROR] --fine-tuned-model and --baseline-model are required."
  usage
  exit 1
fi

if [[ -z "$OUT_JSON" ]]; then
  OUT_JSON="${OUT_MD%.md}.json"
fi

COCO_JSON="${COCO_ROOT}/annotations/instances_val2017.json"
IMAGES_DIR="${COCO_ROOT}/val2017"

if [[ (! -f "$COCO_JSON" || ! -d "$IMAGES_DIR") && "$DOWNLOAD_COCO_IF_MISSING" -eq 1 ]]; then
  echo ">> COCO val2017 data missing. Bootstrapping dataset..."
  bash scripts/fetch_coco2017_val.sh --coco-root "$COCO_ROOT"
fi
if [[ ! -f "$BASELINE_MODEL" && "$DOWNLOAD_BASELINE_IF_MISSING" -eq 1 ]]; then
  echo ">> Baseline model missing. Downloading pretrained EfficientDet-Lite2..."
  bash scripts/fetch_coco2017_val.sh --coco-root "$COCO_ROOT" --with-baseline --baseline-out "$BASELINE_MODEL"
fi

if [[ ! -f "$COCO_JSON" ]]; then
  echo "[ERROR] Missing COCO annotations: $COCO_JSON"
  exit 1
fi
if [[ ! -d "$IMAGES_DIR" ]]; then
  echo "[ERROR] Missing COCO images dir: $IMAGES_DIR"
  exit 1
fi
if [[ ! -f "$FINE_TUNED_MODEL" ]]; then
  echo "[ERROR] Fine-tuned model not found: $FINE_TUNED_MODEL"
  exit 1
fi
if [[ ! -f "$BASELINE_MODEL" ]]; then
  echo "[ERROR] Baseline model not found: $BASELINE_MODEL"
  echo "        Place/download a pretrained EfficientDet-Lite2 TFLite model and pass --baseline-model <path>."
  exit 1
fi
if [[ -n "$CATEGORY_MAP" && ! -f "$CATEGORY_MAP" ]]; then
  echo "[ERROR] Category map not found: $CATEGORY_MAP"
  exit 1
fi

mkdir -p work/reports/val2017_compare
mkdir -p "$(dirname "$OUT_MD")"
mkdir -p "$(dirname "$OUT_JSON")"

FINE_JSON="work/reports/val2017_compare/fine_tuned_eval.json"
BASELINE_JSON="work/reports/val2017_compare/baseline_eval.json"

run_eval() {
  local model_path="$1"
  local out_path="$2"

  echo ">> Evaluating model: $model_path"
  "$MODELMAKER_PYTHON_EXE" - "$COCO_JSON" "$IMAGES_DIR" "$model_path" "$out_path" \
    "$LIMIT_IMAGES" "$SCORE_THRESHOLD" "$NOISE_THRESHOLDS" "$MAX_DETECTIONS" "$CATEGORY_MAP" <<'PY'
from pathlib import Path
import sys

from owli_train.eval.efficientdet_tflite import (
    EfficientDetTFLiteEvalConfigError,
    MissingEfficientDetTFLiteEvalDependenciesError,
    build_eval_efficientdet_tflite_config,
    evaluate_efficientdet_tflite,
)

coco_path = Path(sys.argv[1])
images_dir = Path(sys.argv[2])
model_path = Path(sys.argv[3])
out_path = Path(sys.argv[4])
limit_images = int(sys.argv[5])
score_threshold = float(sys.argv[6])
noise_thresholds_raw = sys.argv[7]
max_detections = int(sys.argv[8])
category_map_arg = sys.argv[9].strip()
category_map_path = Path(category_map_arg) if category_map_arg else None
noise_thresholds = [float(item.strip()) for item in noise_thresholds_raw.split(",") if item.strip()]

try:
    cfg = build_eval_efficientdet_tflite_config(
        coco_path=coco_path,
        images_dir=images_dir,
        model_path=model_path,
        limit_images=limit_images,
        score_threshold=score_threshold,
        noise_thresholds=noise_thresholds,
        max_detections_per_image=max_detections,
        out_path=out_path,
        category_map_path=category_map_path,
    )
    artifacts = evaluate_efficientdet_tflite(cfg)
except (
    EfficientDetTFLiteEvalConfigError,
    MissingEfficientDetTFLiteEvalDependenciesError,
) as exc:
    print(f"ERROR {exc}")
    raise SystemExit(1) from exc

print(f"OK report_json={artifacts.json_report_path}")
print(f"OK report_md={artifacts.markdown_report_path}")
PY
}

run_eval "$FINE_TUNED_MODEL" "$FINE_JSON"
run_eval "$BASELINE_MODEL" "$BASELINE_JSON"

echo ">> Building merged compare report"
"$PYTHON_EXE" - "$FINE_JSON" "$BASELINE_JSON" "$OUT_JSON" "$OUT_MD" "$COCO_JSON" "$IMAGES_DIR" \
  "$FINE_TUNED_MODEL" "$BASELINE_MODEL" "$LIMIT_IMAGES" "$MAX_DETECTIONS" "$SCORE_THRESHOLD" "$NOISE_THRESHOLDS" <<'PY'
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git_value(args: list[str]) -> str:
    try:
        return (
            subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{value:.4f}"


def noise_by_threshold(report: dict) -> dict[float, dict]:
    items = report.get("noise_metrics") or [report["noise_metric"]]
    return {float(item["score_threshold"]): item for item in items}


fine_path = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])
out_json = Path(sys.argv[3])
out_md = Path(sys.argv[4])
coco_json = sys.argv[5]
images_dir = sys.argv[6]
fine_model = sys.argv[7]
baseline_model = sys.argv[8]
limit_images = int(sys.argv[9])
max_detections = int(sys.argv[10])
score_threshold = float(sys.argv[11])
noise_thresholds = [float(item.strip()) for item in sys.argv[12].split(",") if item.strip()]

fine = load_json(fine_path)
baseline = load_json(baseline_path)

fine_noise = noise_by_threshold(fine)
baseline_noise = noise_by_threshold(baseline)

payload = {
    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "branch": git_value(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    "commit": git_value(["git", "rev-parse", "HEAD"]),
    "inputs": {
        "coco_annotations": coco_json,
        "images_dir": images_dir,
        "limit_images": limit_images,
        "max_detections_per_image": max_detections,
        "score_threshold": score_threshold,
        "noise_thresholds": noise_thresholds,
        "fine_tuned_model": fine_model,
        "baseline_model": baseline_model,
    },
    "fine_tuned": fine,
    "baseline": baseline,
}

out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = [
    "# COCO2017 val2017 Evaluation Report (EfficientDet-Lite2)",
    "",
    f"- Date (UTC): `{payload['created_at']}`",
    f"- Branch: `{payload['branch']}`",
    f"- Commit: `{payload['commit']}`",
    f"- COCO annotations: `{coco_json}`",
    f"- COCO images dir: `{images_dir}`",
    f"- Evaluated images: `{limit_images}`",
    f"- Max detections per image: `{max_detections}`",
    f"- Noise thresholds: `{', '.join(f'{t:.2f}' for t in noise_thresholds)}`",
    "",
    "## Models",
    "",
    f"- Fine-tuned: `{fine_model}`",
    f"- Baseline: `{baseline_model}`",
    "",
    "## COCO Metrics (mAP)",
    "",
    "| Model | AP | AP50 | AP75 | AR100 |",
    "|---|---:|---:|---:|---:|",
    f"| Fine-tuned | {fmt(fine['metrics']['AP'])} | {fmt(fine['metrics']['AP50'])} | {fmt(fine['metrics']['AP75'])} | {fmt(fine['metrics']['AR100'])} |",
    f"| Baseline | {fmt(baseline['metrics']['AP'])} | {fmt(baseline['metrics']['AP50'])} | {fmt(baseline['metrics']['AP75'])} | {fmt(baseline['metrics']['AR100'])} |",
    "",
    "## Noise Metrics (FP per 100 images)",
    "",
    "| Model | " + " | ".join(f"@{t:.2f}" for t in noise_thresholds) + " |",
    "|" + "---|" * (len(noise_thresholds) + 1),
    "| Fine-tuned | "
    + " | ".join(
        fmt(float(fine_noise[t]["fp_per_100_images"])) if t in fine_noise else "n/a"
        for t in noise_thresholds
    )
    + " |",
    "| Baseline | "
    + " | ".join(
        fmt(float(baseline_noise[t]["fp_per_100_images"])) if t in baseline_noise else "n/a"
        for t in noise_thresholds
    )
    + " |",
    "",
    f"## Per-Class Aggregate (@{score_threshold:.2f})",
    "",
    "| Model | TP | FP | FN | Precision | Recall |",
    "|---|---:|---:|---:|---:|---:|",
    f"| Fine-tuned | {fine['summary_counts']['tp']} | {fine['summary_counts']['fp']} | {fine['summary_counts']['fn']} | {fmt(fine['summary_counts']['precision'])} | {fmt(fine['summary_counts']['recall'])} |",
    f"| Baseline | {baseline['summary_counts']['tp']} | {baseline['summary_counts']['fp']} | {baseline['summary_counts']['fn']} | {fmt(baseline['summary_counts']['precision'])} | {fmt(baseline['summary_counts']['recall'])} |",
    "",
    "## Raw Reports",
    "",
    f"- Fine-tuned JSON: `{fine_path}`",
    f"- Baseline JSON: `{baseline_path}`",
    f"- Combined JSON: `{out_json}`",
]

out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"OK wrote {out_json}")
print(f"OK wrote {out_md}")
PY

echo "Done."
echo "Combined report: $OUT_MD"
echo "Combined json:   $OUT_JSON"
