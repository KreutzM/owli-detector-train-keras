#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

IMAGE_TAG="${MODELMAKER_DOCKER_IMAGE:-owli-modelmaker-gpu:tf2.8.4}"
DOCKERFILE_PATH="docker/modelmaker-gpu/Dockerfile"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/modelmaker_gpu_docker.sh build [--image <tag>]
  bash scripts/modelmaker_gpu_docker.sh gpu-check [--image <tag>]
  bash scripts/modelmaker_gpu_docker.sh run [--image <tag>] -- <owli args>

Examples:
  bash scripts/modelmaker_gpu_docker.sh build
  bash scripts/modelmaker_gpu_docker.sh gpu-check
  bash scripts/modelmaker_gpu_docker.sh run -- train efficientdet configs/efficientdet_lite2_coco2017.yaml --max-steps 500 --subset-seed 1337 --require-gpu
  bash scripts/modelmaker_gpu_docker.sh run -- eval efficientdet-tflite --coco data/coco2017/annotations/instances_val2017.json --images-dir data/coco2017/val2017 --model work/runs/<run_id>/artifacts/model.tflite --limit-images 128
USAGE
}

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker is not installed or not on PATH."
  exit 1
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

ACTION="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

DOCKER_RUN_BASE=(
  docker run --rm --gpus all
  -v "${REPO_ROOT}:/workspace"
  -w /workspace
)

if command -v id >/dev/null 2>&1; then
  DOCKER_RUN_BASE+=(--user "$(id -u):$(id -g)")
fi

case "$ACTION" in
  build)
    docker build -f "$DOCKERFILE_PATH" -t "$IMAGE_TAG" .
    ;;
  gpu-check)
    "${DOCKER_RUN_BASE[@]}" "$IMAGE_TAG" \
      python3.9 -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
    ;;
  run)
    if [[ $# -eq 0 ]]; then
      echo "[ERROR] Missing owli_train args after 'run --'."
      usage
      exit 1
    fi
    "${DOCKER_RUN_BASE[@]}" "$IMAGE_TAG" python3.9 -m owli_train "$@"
    ;;
  *)
    echo "[ERROR] Unknown action: $ACTION"
    usage
    exit 1
    ;;
esac
