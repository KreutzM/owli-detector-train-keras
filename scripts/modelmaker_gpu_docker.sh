#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

IMAGE_TAG="${MODELMAKER_DOCKER_IMAGE:-owli-modelmaker-gpu:tf2.8.4}"
EXTRA_MOUNTS="${MODELMAKER_DOCKER_EXTRA_MOUNTS:-}"
DOCKERFILE_PATH="docker/modelmaker-gpu/Dockerfile"

DEFAULT_PYTHON_EXE="python3"
HOST_PYTHON_EXE=""
if [[ -x "${REPO_ROOT}/.venv-modelmaker-py39/bin/python" ]]; then
  DEFAULT_PYTHON_EXE="/workspace/.venv-modelmaker-py39/bin/python"
  HOST_PYTHON_EXE="${REPO_ROOT}/.venv-modelmaker-py39/bin/python"
fi
PYTHON_EXE="${MODELMAKER_DOCKER_PYTHON_EXE:-$DEFAULT_PYTHON_EXE}"

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
  MODELMAKER_DOCKER_PYTHON_EXE=/workspace/.venv-modelmaker-py39/bin/python bash scripts/modelmaker_gpu_docker.sh gpu-check
  MODELMAKER_DOCKER_PYTHON_EXE=/workspace/.venv-modelmaker-py39/bin/python MODELMAKER_DOCKER_EXTRA_MOUNTS="$HOME/.local/share/uv:$HOME/.local/share/uv:ro" bash scripts/modelmaker_gpu_docker.sh run -- train efficientdet configs/efficientdet_lite2_coco2017.yaml --max-steps 1 --subset-seed 1337 --require-gpu
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

if [[ -n "$EXTRA_MOUNTS" ]]; then
  IFS=';' read -r -a mount_specs <<<"$EXTRA_MOUNTS"
  for mount_spec in "${mount_specs[@]}"; do
    [[ -z "$mount_spec" ]] && continue
    DOCKER_RUN_BASE+=(-v "$mount_spec")
  done
fi

if [[ -z "$HOST_PYTHON_EXE" && "$PYTHON_EXE" == /workspace/* ]]; then
  candidate="${REPO_ROOT}${PYTHON_EXE#/workspace}"
  if [[ -e "$candidate" ]]; then
    HOST_PYTHON_EXE="$candidate"
  fi
fi

if [[ -n "$HOST_PYTHON_EXE" ]]; then
  uv_root="${HOME}/.local/share/uv"
  resolved_python="$(readlink -f "$HOST_PYTHON_EXE" || true)"
  if [[ -d "$uv_root" && "$resolved_python" == "$uv_root/"* ]]; then
    DOCKER_RUN_BASE+=(-v "${uv_root}:${uv_root}:ro")
  fi
fi

if command -v id >/dev/null 2>&1; then
  DOCKER_RUN_BASE+=(--user "$(id -u):$(id -g)")
fi

case "$ACTION" in
  build)
    docker build -f "$DOCKERFILE_PATH" -t "$IMAGE_TAG" .
    ;;
  gpu-check)
    "${DOCKER_RUN_BASE[@]}" "$IMAGE_TAG" \
      "$PYTHON_EXE" -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
    ;;
  run)
    if [[ $# -eq 0 ]]; then
      echo "[ERROR] Missing owli_train args after 'run --'."
      usage
      exit 1
    fi
    "${DOCKER_RUN_BASE[@]}" "$IMAGE_TAG" "$PYTHON_EXE" -m owli_train "$@"
    ;;
  *)
    echo "[ERROR] Unknown action: $ACTION"
    usage
    exit 1
    ;;
esac
