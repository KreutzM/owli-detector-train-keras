# Android Export Contract (TFLite Detector)

This document defines the runtime contract expected by exported detector `.tflite` models from this repository.

## 1) Input tensor contract

- Tensor rank: 4D NHWC
- Shape: `[1, input_size, input_size, 3]`
- DType: `float32` (default export)
- Channels: RGB
- Resize policy: letterbox/pad to square `input_size` (no aspect distortion)
- Normalization: scale to `[0.0, 1.0]` for parity with current bench/INT8 representative preprocessing

Use `inspect` to verify exact shape/dtype:

```powershell
python -m owli_train inspect tflite --model work\runs\<run_id>\artifacts\detector.tflite
```

## 2) Output tensor contract

Output tensors are model-dependent and should be read by index/name from `inspect`.

For the builtins-first RetinaNet baseline (`arch: retinanet`), outputs are typically:
- boxes tensor: shape `[1, N, 4]` (`xywh`)
- class tensor: shape `[1, N, C]` (per-class scores/logits)

Where:
- `N` = number of anchor predictions
- `C` = number of classes

## 3) Bounding box format

- Export metadata includes `bbox_format` (default `xywh`).
- For this pipeline, boxes are interpreted as `[x, y, width, height]` relative to the letterboxed input frame unless post-processing remaps to original image space.

## 4) Class mapping contract

- Class index `i` maps to `class_names[i]` in:
  - `work\runs\<run_id>\label_map_snapshot.json`
  - `work\runs\<run_id>\artifacts\detector.tflite.meta.json`

This mapping must be kept in sync on Android when rendering labels.

## 5) Compatibility gate

For Android without Flex dependency, enforce:

```powershell
python -m owli_train export tflite --run-dir work\runs\<run_id> --require-builtins-only
```

If export succeeds, metadata contains:
- `android_compat.builtin_ops_only: true`
