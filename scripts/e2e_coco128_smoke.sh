#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python3}"
if ! command -v "$PYTHON_EXE" >/dev/null 2>&1; then
  PYTHON_EXE="python"
fi
MODELMAKER_PYTHON_EXE="${MODELMAKER_PYTHON_EXE:-$PYTHON_EXE}"

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export TF_USE_LEGACY_KERAS="${TF_USE_LEGACY_KERAS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

run_python() {
  echo ">> $PYTHON_EXE $*"
  "$PYTHON_EXE" "$@"
}

run_modelmaker_python() {
  echo ">> $MODELMAKER_PYTHON_EXE $*"
  "$MODELMAKER_PYTHON_EXE" "$@"
}

download_with_python() {
  "$PYTHON_EXE" - "$1" "$2" <<'PY'
import pathlib
import sys
import urllib.request

url = sys.argv[1]
dst = pathlib.Path(sys.argv[2])
dst.parent.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(url) as resp, dst.open("wb") as out:
    out.write(resp.read())
PY
}

extract_with_python() {
  "$PYTHON_EXE" - "$1" "$2" <<'PY'
import pathlib
import sys
import zipfile

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(src, "r") as archive:
    archive.extractall(dst)
PY
}

mkdir -p data work

ZIP_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
ZIP_PATH="data/coco128.zip"
EXTRACT_ROOT="data/coco128_extract"
DATASET_ROOT="data/coco128"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Downloading COCO128 from Ultralytics assets..."
  if command -v curl >/dev/null 2>&1; then
    curl -L "$ZIP_URL" -o "$ZIP_PATH"
  else
    echo "curl not found, using Python download fallback..."
    download_with_python "$ZIP_URL" "$ZIP_PATH"
  fi
else
  echo "Using cached archive: $REPO_ROOT/$ZIP_PATH"
fi

rm -rf "$EXTRACT_ROOT"
mkdir -p "$EXTRACT_ROOT"
if command -v unzip >/dev/null 2>&1; then
  unzip -oq "$ZIP_PATH" -d "$EXTRACT_ROOT"
else
  echo "unzip not found, using Python zipfile fallback..."
  extract_with_python "$ZIP_PATH" "$EXTRACT_ROOT"
fi

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
if [[ "$MODELMAKER_PYTHON_EXE" == "$PYTHON_EXE" ]]; then
  TRAIN_OUTPUT="$("$PYTHON_EXE" -m owli_train train efficientdet --variant lite2 --config configs/efficientdet_lite2_coco128.yaml --max-steps 1 2>&1)"
  TRAIN_EXIT_CODE=$?
else
  TRAIN_OUTPUT="$("$MODELMAKER_PYTHON_EXE" - <<'PY' 2>&1
from pathlib import Path
import sys

from owli_train.training.modelmaker_efficientdet import (
    MissingModelMakerDependenciesError,
    train_efficientdet_from_config,
)

try:
    artifacts = train_efficientdet_from_config(
        config_path=Path("configs/efficientdet_lite2_coco128.yaml"),
        variant="lite2",
        max_steps=1,
    )
except MissingModelMakerDependenciesError as exc:
    print(f"ERROR {exc}")
    raise SystemExit(1) from exc
except Exception as exc:  # pragma: no cover - runtime integration path
    print(f"ERROR {exc}")
    raise SystemExit(1) from exc

print(f"OK run={artifacts.run_id}")
print(f"run_dir: {artifacts.run_dir}")
print(f"tflite: {artifacts.tflite_path}")
PY
)"
  TRAIN_EXIT_CODE=$?
fi
set -e
printf "%s\n" "$TRAIN_OUTPUT"
if [[ $TRAIN_EXIT_CODE -ne 0 ]]; then
  echo "[HINT] Install Model Maker dependencies with: pip install -r requirements/modelmaker.txt"
  if [[ "$MODELMAKER_PYTHON_EXE" != "$PYTHON_EXE" ]]; then
    echo "[HINT] This run uses MODELMAKER_PYTHON_EXE=$MODELMAKER_PYTHON_EXE"
  fi
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
if [[ "$MODELMAKER_PYTHON_EXE" == "$PYTHON_EXE" ]]; then
  run_python -m owli_train inspect tflite --model "$TFLITE_PATH"
else
  run_modelmaker_python - "$TFLITE_PATH" <<'PY'
from pathlib import Path
import sys

from owli_train.export.tflite_export import build_inspect_tflite_config, inspect_tflite

model = Path(sys.argv[1])
cfg = build_inspect_tflite_config(model_path=model)
artifacts = inspect_tflite(cfg)
print(f"OK model={artifacts.model_path}")
print(f"builtin_ops_only: {str(artifacts.builtin_ops_only).lower()}")
print(
    f"operator_names: {', '.join(artifacts.operator_names) if artifacts.operator_names else ''}"
)
print("inputs:")
for item in artifacts.inputs:
    print(f"- {item['name']} shape={item['shape']} dtype={item['dtype']}")
print("outputs:")
for item in artifacts.outputs:
    print(f"- {item['name']} shape={item['shape']} dtype={item['dtype']}")
PY
fi

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
