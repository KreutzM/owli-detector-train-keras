#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python3}"
if ! command -v "$PYTHON_EXE" >/dev/null 2>&1; then
  PYTHON_EXE="python"
fi

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export TF_USE_LEGACY_KERAS="${TF_USE_LEGACY_KERAS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

run_python() {
  echo ">> $PYTHON_EXE $*"
  "$PYTHON_EXE" "$@"
}

if ! command -v curl >/dev/null 2>&1; then
  echo "[ERROR] Missing dependency: curl"
  exit 1
fi
if ! command -v unzip >/dev/null 2>&1; then
  echo "[ERROR] Missing dependency: unzip"
  exit 1
fi

mkdir -p data work

ZIP_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
ZIP_PATH="data/coco128.zip"
EXTRACT_ROOT="data/coco128_extract"
DATASET_ROOT="data/coco128"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Downloading COCO128 from Ultralytics assets..."
  curl -L "$ZIP_URL" -o "$ZIP_PATH"
else
  echo "Using cached archive: $REPO_ROOT/$ZIP_PATH"
fi

rm -rf "$EXTRACT_ROOT"
mkdir -p "$EXTRACT_ROOT"
unzip -oq "$ZIP_PATH" -d "$EXTRACT_ROOT"

resolve_root() {
  local root="$1"
  if [[ -d "$root/images" && -d "$root/labels" ]]; then
    printf "%s\n" "$root"
    return 0
  fi
  while IFS= read -r candidate; do
    if [[ -d "$candidate/images" && -d "$candidate/labels" ]]; then
      printf "%s\n" "$candidate"
      return 0
    fi
  done < <(find "$root" -type d)
  return 1
}

RESOLVED_EXTRACTED_ROOT="$(resolve_root "$EXTRACT_ROOT")" || {
  echo "[ERROR] Could not locate COCO128 YOLO root under: $EXTRACT_ROOT"
  exit 1
}

rm -rf "$DATASET_ROOT"
mkdir -p "$DATASET_ROOT"
cp -a "$RESOLVED_EXTRACTED_ROOT/." "$DATASET_ROOT/"

DATA_YAML_PATH="$DATASET_ROOT/coco128.yaml"
if [[ ! -f "$DATA_YAML_PATH" ]]; then
  echo "No dataset yaml found. Generating fallback class-name mapping from labels..."
  "$PYTHON_EXE" - <<'PY'
from pathlib import Path

dataset_root = Path("data/coco128")
labels_root = dataset_root / "labels"
class_ids = set()

for label_file in labels_root.rglob("*.txt"):
    for raw in label_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        try:
            class_ids.add(int(parts[0]))
        except (IndexError, ValueError):
            continue

if not class_ids:
    raise SystemExit(f"Could not infer class IDs from label files under: {labels_root}")

max_id = max(class_ids)
yaml_lines = ["names:"]
yaml_lines.extend([f"  {idx}: class_{idx}" for idx in range(max_id + 1)])
(dataset_root / "coco128.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
PY
fi

removed_dangling_labels="$("$PYTHON_EXE" - <<'PY'
from pathlib import Path

dataset_root = Path("data/coco128")
images_root = dataset_root / "images"
labels_root = dataset_root / "labels"
suffixes = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
removed = 0

for label_file in labels_root.rglob("*.txt"):
    rel = label_file.relative_to(labels_root)
    stem = rel.with_suffix("")
    has_image = any((images_root / stem).with_suffix(ext).is_file() for ext in suffixes)
    if not has_image:
        label_file.unlink()
        removed += 1

print(removed)
PY
)"

if [[ "$removed_dangling_labels" != "0" ]]; then
  echo "Removed $removed_dangling_labels label file(s) without matching image."
fi

COCO_OUT="work/datasets/coco128/instances.json"
SPLITS_DIR="work/datasets/coco128/splits"
SPLITS_JSON="work/datasets/coco128/splits/splits.json"
CSV_OUT="work/datasets/coco128/modelmaker.csv"

run_python -m owli_train dataset import yolo \
  --yolo-dir data/coco128 \
  --data-yaml data/coco128/coco128.yaml \
  --out "$COCO_OUT"

run_python -m owli_train dataset split \
  --coco "$COCO_OUT" \
  --out-dir "$SPLITS_DIR" \
  --seed 1337

run_python -m owli_train dataset export modelmaker-csv \
  --coco "$COCO_OUT" \
  --images-dir data/coco128/images \
  --splits-json "$SPLITS_JSON" \
  --out "$CSV_OUT"

set +e
TRAIN_OUTPUT="$("$PYTHON_EXE" -m owli_train train efficientdet --variant lite2 --config configs/efficientdet_lite2_coco128.yaml --max-steps 1 2>&1)"
TRAIN_EXIT_CODE=$?
set -e
printf "%s\n" "$TRAIN_OUTPUT"
if [[ $TRAIN_EXIT_CODE -ne 0 ]]; then
  echo "[HINT] Install Model Maker dependencies with: pip install -r requirements/modelmaker.txt"
  echo "EfficientDet smoke training failed (exit=$TRAIN_EXIT_CODE)."
  exit "$TRAIN_EXIT_CODE"
fi

RUN_DIR_LINE="$(printf "%s\n" "$TRAIN_OUTPUT" | grep -F "run_dir:" | tail -n 1 || true)"
if [[ -z "$RUN_DIR_LINE" ]]; then
  echo "[ERROR] Could not parse run_dir from training output."
  exit 1
fi
RUN_DIR="${RUN_DIR_LINE#run_dir: }"
RUN_DIR="${RUN_DIR#"${RUN_DIR%%[![:space:]]*}"}"
RUN_DIR="${RUN_DIR%"${RUN_DIR##*[![:space:]]}"}"

TFLITE_PATH="$RUN_DIR/artifacts/model.tflite"
if [[ ! -f "$TFLITE_PATH" ]]; then
  echo "[ERROR] Expected TFLite artifact not found: $TFLITE_PATH"
  exit 1
fi

echo "Verified TFLite artifact: $TFLITE_PATH"
echo "Inspecting exported TFLite model..."
run_python -m owli_train inspect tflite --model "$TFLITE_PATH"

echo ""
echo "E2E smoke completed successfully."
echo "run_dir: $RUN_DIR"
echo "tflite: $TFLITE_PATH"
echo ""
echo "Generated paths:"
echo "  data/coco128.zip"
echo "  data/coco128/"
echo "  work/datasets/coco128/"
echo "  work/runs/<run_id>/"
echo ""
echo "Cleanup:"
echo "  rm -rf data/coco128 data/coco128_extract work/datasets/coco128"
echo "  rm -rf work/runs/<run_id>"
