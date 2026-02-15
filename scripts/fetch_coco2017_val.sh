#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

COCO_ROOT="data/coco2017"
WITH_BASELINE=0
BASELINE_OUT="work/models/efficientdet_lite2_baseline.tflite"
FORCE=0

VAL_ZIP_URL="http://images.cocodataset.org/zips/val2017.zip"
ANN_ZIP_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
BASELINE_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_efficientdet_lite2_detection_metadata_1.tflite"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/fetch_coco2017_val.sh [options]

Options:
  --coco-root <path>      Dataset root (default: data/coco2017)
  --with-baseline         Also download baseline EfficientDet-Lite2 model
  --baseline-out <path>   Baseline model output path (default: work/models/efficientdet_lite2_baseline.tflite)
  --force                 Re-download and re-extract files
  -h, --help              Show help
USAGE
}

download_with_python() {
  local url="$1"
  local dst="$2"
  python3 - "$url" "$dst" <<'PY'
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
  local src="$1"
  local dst="$2"
  python3 - "$src" "$dst" <<'PY'
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

download_file() {
  local url="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$dst"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -O "$dst" "$url"
    return
  fi
  download_with_python "$url" "$dst"
}

extract_zip() {
  local src="$1"
  local dst="$2"
  mkdir -p "$dst"
  if command -v unzip >/dev/null 2>&1; then
    unzip -oq "$src" -d "$dst"
    return
  fi
  extract_with_python "$src" "$dst"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --coco-root)
      COCO_ROOT="$2"
      shift 2
      ;;
    --with-baseline)
      WITH_BASELINE=1
      shift
      ;;
    --baseline-out)
      BASELINE_OUT="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
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

VAL_ZIP_PATH="${COCO_ROOT}/val2017.zip"
ANN_ZIP_PATH="${COCO_ROOT}/annotations_trainval2017.zip"
VAL_DIR="${COCO_ROOT}/val2017"
ANN_FILE="${COCO_ROOT}/annotations/instances_val2017.json"

if [[ "$FORCE" -eq 1 ]]; then
  rm -rf "$VAL_DIR"
  rm -f "$ANN_FILE"
fi

if [[ ! -f "$VAL_ZIP_PATH" || "$FORCE" -eq 1 ]]; then
  echo ">> Downloading val2017.zip"
  download_file "$VAL_ZIP_URL" "$VAL_ZIP_PATH"
else
  echo ">> Using cached: $VAL_ZIP_PATH"
fi

if [[ ! -d "$VAL_DIR" || -z "$(find "$VAL_DIR" -maxdepth 1 -type f -name '*.jpg' 2>/dev/null | head -n 1)" ]]; then
  echo ">> Extracting val2017 images"
  extract_zip "$VAL_ZIP_PATH" "$COCO_ROOT"
fi

if [[ ! -f "$ANN_ZIP_PATH" || "$FORCE" -eq 1 ]]; then
  echo ">> Downloading annotations_trainval2017.zip"
  download_file "$ANN_ZIP_URL" "$ANN_ZIP_PATH"
else
  echo ">> Using cached: $ANN_ZIP_PATH"
fi

if [[ ! -f "$ANN_FILE" ]]; then
  echo ">> Extracting COCO annotations"
  extract_zip "$ANN_ZIP_PATH" "$COCO_ROOT"
fi

if [[ "$WITH_BASELINE" -eq 1 ]]; then
  if [[ ! -f "$BASELINE_OUT" || "$FORCE" -eq 1 ]]; then
    echo ">> Downloading baseline model"
    download_file "$BASELINE_URL" "$BASELINE_OUT"
  else
    echo ">> Using cached: $BASELINE_OUT"
  fi
fi

if [[ ! -f "$ANN_FILE" ]]; then
  echo "[ERROR] Missing annotations after setup: $ANN_FILE"
  exit 1
fi
if [[ ! -d "$VAL_DIR" ]]; then
  echo "[ERROR] Missing val image directory after setup: $VAL_DIR"
  exit 1
fi

jpg_count="$(find "$VAL_DIR" -maxdepth 1 -type f -name '*.jpg' | wc -l | tr -d ' ')"
echo "OK coco_root=$COCO_ROOT"
echo "OK images_dir=$VAL_DIR (jpg=$jpg_count)"
echo "OK annotations=$ANN_FILE"
if [[ "$WITH_BASELINE" -eq 1 ]]; then
  echo "OK baseline_model=$BASELINE_OUT"
fi
