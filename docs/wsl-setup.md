# WSL2 Setup (Ubuntu)

## Where to clone

Use the Linux filesystem (for example `~/src/owli-detector-train-keras`) instead of `/mnt/c/...`.

Paraphrased guidance from Microsoft WSL docs: keeping project files inside the Linux filesystem improves filesystem performance and tool reliability compared with working from mounted Windows drives.

Reference:
- https://learn.microsoft.com/windows/wsl/filesystems

## Environment setup

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements/dev.txt
```

Optional training dependencies:

```bash
pip install -r requirements/keras.txt
```

For the EfficientDet Model Maker backend, prefer a separate venv:

```bash
deactivate  # if needed
python3.9 -m venv .venv-modelmaker
source .venv-modelmaker/bin/activate
python -m pip install --upgrade pip
pip install -r requirements/modelmaker.txt
```

Notes:
- `scripts/e2e_coco128_smoke.sh` has Python fallbacks for download/extract, so `curl`/`unzip` are optional.
- Model Maker currently targets a legacy stack. Use Python 3.9 for `.venv-modelmaker`.
- The smoke script supports split interpreters via `MODELMAKER_PYTHON_EXE`:
  `MODELMAKER_PYTHON_EXE=.venv-modelmaker-py39/bin/python bash scripts/e2e_coco128_smoke.sh`
- Before long runs, verify GPU visibility in that interpreter:
  `.venv-modelmaker/bin/python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"`
- Use `--require-gpu` with `train efficientdet` to fail fast if training would run on CPU.

## Docker GPU fallback (legacy Model Maker)

If the local Model Maker venv cannot initialize CUDA (`tf.config.list_physical_devices('GPU') == []`), run EfficientDet through the project Docker image.

WSL/bash:

```bash
bash scripts/modelmaker_gpu_docker.sh build
bash scripts/modelmaker_gpu_docker.sh gpu-check
bash scripts/modelmaker_gpu_docker.sh run -- train efficientdet configs/efficientdet_lite2_coco2017.yaml --max-steps 500 --subset-seed 1337 --require-gpu
```

The bash script auto-picks `/workspace/.venv-modelmaker-py39/bin/python` if `.venv-modelmaker-py39` exists in the repo.

If your selected container Python is a mounted venv path (for example a py39 venv that symlinks into `~/.local/share/uv`), set:

```bash
export MODELMAKER_DOCKER_PYTHON_EXE=/workspace/.venv-modelmaker-py39/bin/python
export MODELMAKER_DOCKER_EXTRA_MOUNTS="$HOME/.local/share/uv:$HOME/.local/share/uv:ro"
bash scripts/modelmaker_gpu_docker.sh gpu-check
```

PowerShell:

```powershell
.\scripts\modelmaker_gpu_docker.ps1 build
.\scripts\modelmaker_gpu_docker.ps1 gpu-check
.\scripts\modelmaker_gpu_docker.ps1 run train efficientdet configs\efficientdet_lite2_coco2017.yaml --max-steps 500 --subset-seed 1337 --require-gpu
```

## TensorFlow GPU note

For TensorFlow `>=2.11`, GPU-enabled workflows are typically run through WSL2 on Windows. Native Windows TensorFlow installs are commonly CPU-only for these versions.

## WSL smoke entry point

```bash
bash scripts/e2e_coco128_smoke.sh
```

The script only writes to:
- `data/`
- `work/`
