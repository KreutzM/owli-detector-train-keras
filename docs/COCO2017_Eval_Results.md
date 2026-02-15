# COCO2017 EfficientDet-Lite2 Results (WSL2 GPU)

Date: 2026-02-15
Branch: `feat/m9-eval-efficientdet-tflite`
Commit (start of run): `fec89fe`

## Environment summary

- Host: `Linux Michael 6.6.87.2-microsoft-standard-WSL2 x86_64`
- GPU (`nvidia-smi`): NVIDIA GeForce RTX 3060, Driver `576.80`, CUDA `12.9`
- Main tool env: `.venv` (`Python 3.10.12`)
- Model Maker env: `.venv-modelmaker-py39/bin/python`
- TensorFlow in Model Maker env: `2.8.4`
- `tf.config.list_physical_devices('GPU')`: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

## Commands executed

Training pipeline:

```bash
tmux new-session -d -s coco2017_train \
  'cd /home/michael/src/train && bash ./work/run_coco2017_train_pipeline.sh |& tee -a work/logs/coco2017_train_pipeline.log'
```

Notes:
- The pipeline downloaded/extracted COCO2017 train + annotations, removed 2 invalid-width/height boxes, exported Model Maker CSV, then trained EfficientDet-Lite2 with `max_steps=5000`.

Eval (Model Maker env; direct function call used in this historical run):

```bash
.venv-modelmaker-py39/bin/python - <<'PY'
from pathlib import Path
from owli_train.eval.efficientdet_tflite import build_eval_efficientdet_tflite_config, evaluate_efficientdet_tflite
run_dir = Path('work/runs/20260215-005450-coco2017-train')
for limit in (20, 128):
    cfg = build_eval_efficientdet_tflite_config(
        coco_path=Path('work/datasets/coco2017/instances_train2017.clean.json'),
        images_dir=Path('data/coco2017/images/train2017'),
        model_path=run_dir / 'artifacts/model.tflite',
        limit_images=limit,
        score_threshold=0.3,
        max_detections_per_image=100,
        out_path=run_dir / f'reports/eval_efficientdet_tflite_limit{limit}.json',
        category_map_path=None,
    )
    evaluate_efficientdet_tflite(cfg)
PY
```

Golden sample:

```bash
.venv-modelmaker-py39/bin/python - <<'PY'
from pathlib import Path
from owli_train.golden.detect import build_golden_detect_config, generate_golden_detect
cfg = build_golden_detect_config(
    model_path=Path('work/runs/20260215-005450-coco2017-train/artifacts/model.tflite'),
    image_path=Path('data/coco2017/images/train2017/000000000009.jpg'),
    out_path=Path('work/runs/20260215-005450-coco2017-train/reports/golden_coco2017_first.json'),
    score_threshold=0.3,
    max_results=20,
)
generate_golden_detect(cfg)
PY
```

## Run directory

- `work/runs/20260215-005450-coco2017-train`

## Artifact verification

Under `work/runs/20260215-005450-coco2017-train/artifacts`:
- `model.tflite`
- `model.json`
- `labels.txt`
- `class_names.json`

## Training outcome

- Completed full run: `5000/5000` steps
- Final train metrics:
  - `loss=0.9530`
  - `det_loss=0.9336`
  - `cls_loss=0.6557`
  - `box_loss=0.0056`
- Final validation metrics:
  - `val_loss=0.8508`
  - `val_det_loss=0.8312`
  - `val_cls_loss=0.7052`
  - `val_box_loss=0.0025`

## TFLite inspect / contract summary

- Model: `work/runs/20260215-005450-coco2017-train/artifacts/model.tflite`
- `builtin_ops_only`: `true`
- Operators: `QUANTIZE, CONV_2D, DEPTHWISE_CONV_2D, ADD, MAX_POOL_2D, RESIZE_NEAREST_NEIGHBOR, RESHAPE, CONCATENATION, LOGISTIC, DEQUANTIZE, TFLite_Detection_PostProcess`
- Input preprocessing/runtime:
  - `resize_policy=letterbox_square`
  - `input_shape=[1, 448, 448, 3]`
  - `input_dtype=uint8`
  - `normalization=none`
  - `bbox_format=xywh`, coordinates in absolute pixels

## Eval results (`score_threshold=0.3`)

Reports:
- `work/runs/20260215-005450-coco2017-train/reports/eval_efficientdet_tflite_limit20.json`
- `work/runs/20260215-005450-coco2017-train/reports/eval_efficientdet_tflite_limit20.md`
- `work/runs/20260215-005450-coco2017-train/reports/eval_efficientdet_tflite_limit128.json`
- `work/runs/20260215-005450-coco2017-train/reports/eval_efficientdet_tflite_limit128.md`

`limit=20`:
- AP: `0.2574`, AP50: `0.3089`, AP75: `0.3023`
- AR100: `0.2593`
- Detections kept: `48`
- Noise metric: `FP/100 images = 115.000` (FP=`23`)
- Aggregate: Precision `0.5208`, Recall `0.2809`

`limit=128`:
- AP: `0.1068`, AP50: `0.1500`, AP75: `0.1185`
- AR100: `0.1188`
- Detections kept: `428`
- Noise metric: `FP/100 images = 128.125` (FP=`164`)
- Aggregate: Precision `0.6168`, Recall `0.2812`

## Golden sample

- Path: `work/runs/20260215-005450-coco2017-train/reports/golden_coco2017_first.json`
- Source image (deterministic first file): `data/coco2017/images/train2017/000000000009.jpg`
- Detections in sample: `3`

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
  "detections": [
    {
      "class_name": "bowl",
      "score": 0.359375
    }
  ]
}
```

## Notes

- At the time of this run, the Python 3.9 CLI path had a Typer type-hint incompatibility. This has since been fixed (`efbba06`), so direct CLI invocation is now supported again.
