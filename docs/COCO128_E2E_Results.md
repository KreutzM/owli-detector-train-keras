# COCO128 E2E Results (WSL2 GPU)

Date: 2026-02-14
Branch: `feat/m9-eval-efficientdet-tflite`
Commit (start of run): `fec89fe`

## Environment summary

- Host: `Linux Michael 6.6.87.2-microsoft-standard-WSL2 x86_64`
- GPU (`nvidia-smi`): NVIDIA GeForce RTX 3060 detected (CUDA 12.9, driver 576.80)
- Main tool env: `.venv` (`Python 3.10.12`)
- Model Maker env selected: `.venv-modelmaker-py39/bin/python`
- TensorFlow in Model Maker env: `2.8.4`
- `tf.config.list_physical_devices('GPU')`: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

## Commands executed

Preflight:

```bash
uname -a
nvidia-smi
source .venv/bin/activate && python --version
.venv-modelmaker-py39/bin/python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Dataset conversion (main env):

```bash
source .venv/bin/activate
python -m owli_train dataset import yolo --yolo-dir data/coco128 --out work/datasets/coco128/instances.json
python -m owli_train dataset split --coco work/datasets/coco128/instances.json --out-dir work/splits/coco128 --seed 1337
python -m owli_train dataset export modelmaker-csv --coco work/datasets/coco128/instances.json --images-dir data/coco128/images --splits-json work/splits/coco128/splits.json --out work/datasets/coco128/modelmaker.csv
```

Training (Model Maker env, equivalent backend function call due Python 3.9 CLI issue):

```bash
.venv-modelmaker-py39/bin/python - <<'PY'
from pathlib import Path
from owli_train.training.modelmaker_efficientdet import train_efficientdet_from_config
art = train_efficientdet_from_config(config_path=Path('configs/efficientdet_lite2_coco128.yaml'), max_steps=500)
print(art.run_dir)
PY
```

Inspect/eval/golden (Model Maker env):

```bash
# inspect (module function)
.venv-modelmaker-py39/bin/python - <<'PY'
from pathlib import Path
from owli_train.export.tflite_export import build_inspect_tflite_config, inspect_tflite
art = inspect_tflite(build_inspect_tflite_config(model_path=Path('work/runs/20260214-211454/artifacts/model.tflite')))
print(art.builtin_ops_only)
PY

# eval limit 20 + 128 (module function)
.venv-modelmaker-py39/bin/python - <<'PY'
from pathlib import Path
from owli_train.eval.efficientdet_tflite import build_eval_efficientdet_tflite_config, evaluate_efficientdet_tflite
for limit in (20, 128):
    cfg = build_eval_efficientdet_tflite_config(
        coco_path=Path('work/datasets/coco128/instances.json'),
        images_dir=Path('data/coco128/images'),
        model_path=Path('work/runs/20260214-211454/artifacts/model.tflite'),
        limit_images=limit,
        score_threshold=0.3,
        max_detections_per_image=100,
        out_path=Path(f'work/runs/20260214-211454/reports/eval_efficientdet_tflite_{limit}.json'),
        category_map_path=None,
    )
    evaluate_efficientdet_tflite(cfg)
PY

# golden sample
.venv-modelmaker-py39/bin/python - <<'PY'
from pathlib import Path
from owli_train.golden.detect import build_golden_detect_config, generate_golden_detect
cfg = build_golden_detect_config(
    model_path=Path('work/runs/20260214-211454/artifacts/model.tflite'),
    image_path=Path('data/coco128/images/train2017/000000000009.jpg'),
    out_path=Path('work/runs/20260214-211454/reports/golden_coco128.json'),
    score_threshold=0.3,
    max_results=20,
)
generate_golden_detect(cfg)
PY
```

## Run directory

- `work/runs/20260214-211454`

## Artifact verification

Under `work/runs/20260214-211454/artifacts`:
- `model.tflite`
- `model.json` (Model Maker metadata JSON)
- `labels.txt`
- `class_names.json`

## TFLite inspect / contract summary

- Model: `work/runs/20260214-211454/artifacts/model.tflite`
- `builtin_ops_only`: `true`
- Operators: `QUANTIZE, CONV_2D, DEPTHWISE_CONV_2D, ADD, MAX_POOL_2D, RESIZE_NEAREST_NEIGHBOR, RESHAPE, CONCATENATION, LOGISTIC, DEQUANTIZE, TFLite_Detection_PostProcess`
- Input tensor (interpreter):
  - name: `serving_default_images:0`
  - shape: `[1, 448, 448, 3]`
  - dtype: `uint8`
  - quantization: `(0.0078125, 127)`
- Output tensors (interpreter):
  - `StatefulPartitionedCall:1` shape `[1, 25]` float32
  - `StatefulPartitionedCall:3` shape `[1, 25, 4]` float32
  - `StatefulPartitionedCall:0` shape `[1]` float32
  - `StatefulPartitionedCall:2` shape `[1, 25]` float32

## Eval results (`score_threshold=0.3`)

Report files:
- `work/runs/20260214-211454/reports/eval_efficientdet_tflite_20.json`
- `work/runs/20260214-211454/reports/eval_efficientdet_tflite_20.md`
- `work/runs/20260214-211454/reports/eval_efficientdet_tflite_128.json`
- `work/runs/20260214-211454/reports/eval_efficientdet_tflite_128.md`

Metrics (limit 20):
- `AP=0.0000`, `AP50=0.0000`, `AP75=0.0000`, `AR100=0.0000`
- Noise metric: `FP/100 images = 0.0000`
- Detections kept: `0`

Metrics (limit 128; dataset contains 126 images):
- `AP=0.0000`, `AP50=0.0000`, `AP75=0.0000`, `AR100=0.0000`
- Noise metric: `FP/100 images = 0.0000`
- Detections kept: `0`

## Golden sample

- Path: `work/runs/20260214-211454/reports/golden_coco128.json`
- Source image (deterministic first file): `data/coco128/images/train2017/000000000009.jpg`
- Detections in sample: `0`

Snippet:

```json
{
  "contract": {
    "score_threshold": 0.3,
    "max_results": 20,
    "input_preprocessing": {
      "resize_policy": "letterbox_square",
      "target_size": 448,
      "input_shape": [1, 448, 448, 3],
      "input_dtype": "uint8",
      "normalization": "none"
    }
  },
  "detections": []
}
```

## Notes / deviations

- `python -m owli_train ...` under Python 3.9 failed in Typer signature evaluation (`TypeError` on `| None` hints). For this run, Model Maker steps were executed via direct module function calls in the same interpreter.
- Installing `requirements/eval.txt` into the Model Maker env upgraded `numpy` to `2.0.2` (incompatible with TF 2.8 / Model Maker). `numpy` was pinned back to `1.23.3` before eval/golden execution.
- Runtime bug found/fixed during this run: empty detection lists caused `pycocotools.loadRes([])` `IndexError`. Fix implemented with zero-metric fallback and unit test.
