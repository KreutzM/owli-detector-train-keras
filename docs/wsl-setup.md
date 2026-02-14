# WSL2 Setup (Ubuntu)

## Where to clone

Use the Linux filesystem (for example `~/src/owli-detector-train-keras`) instead of `/mnt/c/...`.

Paraphrased guidance from Microsoft WSL docs: keeping project files inside the Linux filesystem improves filesystem performance and tool reliability compared with working from mounted Windows drives.

Reference:
- https://learn.microsoft.com/windows/wsl/filesystems

## Environment setup

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip curl unzip

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
python3 -m venv .venv-modelmaker
source .venv-modelmaker/bin/activate
python -m pip install --upgrade pip
pip install -r requirements/modelmaker.txt
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
