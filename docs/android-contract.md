# Android Detector Contract (EfficientDet TFLite)

This document defines the stable detector inference contract for Android integration and for `python -m owli_train golden detect`.

## Scope

Applies to exported EfficientDet TFLite models used by:
- `python -m owli_train eval efficientdet-tflite`
- `python -m owli_train golden detect`
- Android offline inference code consuming the same model artifacts

## Input Preprocessing

Required preprocessing steps:
- Color space: `RGB`
- Resize policy: `letterbox_square` (preserve aspect ratio, pad to square)
- Target size: model input size from the TFLite input tensor (`[1, H, W, 3]`, normally square)
- Pad value: `0` (black)
- Tensor dtype handling:
  - `float32`: convert image to float and normalize to `[0, 1]`
  - `uint8`: keep pixel values in `[0, 255]`
  - `int8`: quantize from normalized `[0, 1]` using input quantization scale/zero-point

The golden sample JSON includes the exact preprocessing fields used at runtime under `contract.input_preprocessing`.

## Output Bounding Boxes

Detection boxes in the contract are:
- Format: `xywh`
- Coordinates: `absolute_pixels`
- Coordinate frame: original image pixels (after unletterbox remap)

Each detection entry uses:
- `bbox`: `[x, y, width, height]`
- `bbox_format`: `xywh`
- `coordinates`: `absolute_pixels`

## Class Label Ordering

Class index -> class name mapping source priority:
1. `model.tflite.meta.json` -> `class_names`
2. `labels.txt` next to the model
3. `class_names.json` next to the model
4. Fallback name: `class_<index>`

Android must use the same ordering source as the generated golden sample (`contract.class_labels_source`).

## Score Threshold and maxResults

Runtime behavior:
- Detections are sorted by descending score.
- Detections below `score_threshold` are dropped.
- At most `maxResults` (or `max_detections_per_image` in eval) are returned.

Defaults:
- `golden detect`: `--score-threshold 0.3`, `--max-results 20`
- `eval efficientdet-tflite`: `--score-threshold 0.3`, `--max-detections-per-image 100`

## Golden Sample Contract

Generate:

```powershell
python -m owli_train golden detect --model <model.tflite> --image <path> --out work\golden\sample.json
```

WSL:

```bash
python -m owli_train golden detect --model <model.tflite> --image <path> --out work/golden/sample.json
```

The JSON includes:
- preprocessing params used
- top detections (`class_name`, `score`, `bbox`)
- model metadata snapshot (if `*.meta.json` exists)
- TFLite inspect snapshot (ops + IO tensors)

Use this file as the Android regression fixture for parser/preprocessing parity checks.
