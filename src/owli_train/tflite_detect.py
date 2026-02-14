from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from owli_train.export.tflite_export import (
    MissingTFLiteDependenciesError,
    build_inspect_tflite_config,
    inspect_tflite,
)


class TFLiteDetectError(RuntimeError):
    """Base class for TFLite detector runtime failures."""


@dataclass(frozen=True)
class TFLitePreprocessSpec:
    input_size: int
    input_shape: list[int]
    input_dtype: str
    normalization: str
    resize_policy: str = "letterbox_square"
    color_space: str = "RGB"
    bbox_format: str = "xywh"
    coordinates: str = "absolute_pixels"
    pad_value: int = 0


@dataclass
class TFLiteRuntime:
    interpreter: Any
    input_index: int
    output_indices: list[int]
    input_dtype: Any
    input_quantization: tuple[float, int]
    preprocess: TFLitePreprocessSpec
    builtin_ops_only: bool
    operator_names: list[str]
    inspect_inputs: list[dict[str, Any]]
    inspect_outputs: list[dict[str, Any]]


@dataclass(frozen=True)
class TFLiteDetection:
    class_index: int
    score: float
    bbox_xywh: tuple[float, float, float, float]


@dataclass(frozen=True)
class TFLiteLabelMap:
    class_names: list[str]
    source: str


def ensure_tflite_runtime_dependencies() -> Any:
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - runtime environment specific
        raise MissingTFLiteDependenciesError(
            "TensorFlow Lite runtime is required. Install with: pip install -r "
            "requirements\\modelmaker.txt"
        ) from exc
    return tf


def load_tflite_metadata(model_path: Path) -> dict[str, Any] | None:
    meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_tflite_label_map(model_path: Path, metadata: dict[str, Any] | None) -> TFLiteLabelMap:
    if isinstance(metadata, dict):
        class_names = metadata.get("class_names")
        if isinstance(class_names, list) and class_names:
            return TFLiteLabelMap(
                class_names=[str(v) for v in class_names], source="meta.class_names"
            )

    labels_path = model_path.parent / "labels.txt"
    if labels_path.is_file():
        labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines()]
        labels = [value for value in labels if value]
        if labels:
            return TFLiteLabelMap(class_names=labels, source="labels.txt")

    class_names_json = model_path.parent / "class_names.json"
    if class_names_json.is_file():
        try:
            payload = json.loads(class_names_json.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, list) and payload:
            return TFLiteLabelMap(class_names=[str(v) for v in payload], source="class_names.json")

    return TFLiteLabelMap(class_names=[], source="class_index_fallback")


def _coerce_input_size(shape: list[int], metadata: dict[str, Any] | None) -> int:
    h = shape[1] if len(shape) > 1 else -1
    w = shape[2] if len(shape) > 2 else -1
    if h > 0 and w > 0:
        return max(h, w)

    if isinstance(metadata, dict):
        meta_size = metadata.get("input_size")
        if isinstance(meta_size, int) and meta_size > 0:
            return meta_size

    return 640


def _normalize_shape(raw_shape: Any) -> list[int]:
    if not isinstance(raw_shape, (list, tuple, np.ndarray)):
        return [1, 640, 640, 3]
    return [int(v) for v in raw_shape]


def create_tflite_runtime(
    *,
    model_path: Path,
    tf: Any | None = None,
) -> TFLiteRuntime:
    resolved_tf = tf if tf is not None else ensure_tflite_runtime_dependencies()
    inspect_artifacts = inspect_tflite(build_inspect_tflite_config(model_path=model_path))
    metadata = load_tflite_metadata(model_path)

    interpreter = resolved_tf.lite.Interpreter(model_path=str(model_path))
    input_details = interpreter.get_input_details()
    if not input_details:
        raise TFLiteDetectError("TFLite model has no input tensor.")
    input_info = input_details[0]

    shape = _normalize_shape(input_info.get("shape"))
    if len(shape) != 4:
        raise TFLiteDetectError(f"Expected 4D NHWC input, got shape={shape}.")

    input_size = _coerce_input_size(shape, metadata)
    resized_shape = list(shape)
    if resized_shape[0] <= 0:
        resized_shape[0] = 1
    if resized_shape[1] <= 0:
        resized_shape[1] = input_size
    if resized_shape[2] <= 0:
        resized_shape[2] = input_size
    if resized_shape[3] <= 0:
        resized_shape[3] = 3

    needs_resize = resized_shape != shape
    if needs_resize:
        interpreter.resize_tensor_input(int(input_info["index"]), resized_shape)

    try:
        interpreter.allocate_tensors()
    except RuntimeError as exc:  # pragma: no cover - runtime/environment specific
        raise TFLiteDetectError(
            "Failed to initialize TFLite interpreter. If this model uses SELECT_TF_OPS, ensure "
            "the TensorFlow Lite runtime supports it."
        ) from exc

    refreshed_input = interpreter.get_input_details()[0]
    refreshed_shape = _normalize_shape(refreshed_input.get("shape"))
    if len(refreshed_shape) != 4:
        raise TFLiteDetectError(f"Expected 4D NHWC input after allocate, got {refreshed_shape}.")

    dtype = refreshed_input["dtype"]
    dtype_name = np.dtype(dtype).name
    normalization = "[0,1]" if dtype_name == "float32" else "none"

    output_details = interpreter.get_output_details()
    output_indices = [int(item["index"]) for item in output_details]
    input_quantization = refreshed_input.get("quantization") or (0.0, 0)
    q_scale = float(input_quantization[0]) if len(input_quantization) > 0 else 0.0
    q_zero = int(input_quantization[1]) if len(input_quantization) > 1 else 0

    preprocess = TFLitePreprocessSpec(
        input_size=max(int(refreshed_shape[1]), int(refreshed_shape[2])),
        input_shape=refreshed_shape,
        input_dtype=dtype_name,
        normalization=normalization,
    )
    return TFLiteRuntime(
        interpreter=interpreter,
        input_index=int(refreshed_input["index"]),
        output_indices=output_indices,
        input_dtype=dtype,
        input_quantization=(q_scale, q_zero),
        preprocess=preprocess,
        builtin_ops_only=inspect_artifacts.builtin_ops_only,
        operator_names=inspect_artifacts.operator_names,
        inspect_inputs=inspect_artifacts.inputs,
        inspect_outputs=inspect_artifacts.outputs,
    )


def letterbox_image(
    *,
    image_path: Path,
    input_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    with Image.open(image_path) as img:
        image = img.convert("RGB")

    orig_w, orig_h = image.size
    scale = min(input_size / float(orig_w), input_size / float(orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (input_size, input_size), color=(0, 0, 0))
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.asarray(canvas, dtype=np.uint8)
    meta = {
        "orig_w": float(orig_w),
        "orig_h": float(orig_h),
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "input_size": float(input_size),
    }
    return arr, meta


def _to_input_tensor(
    *,
    image: np.ndarray,
    input_dtype: Any,
    input_quantization: tuple[float, int],
) -> np.ndarray:
    dtype_name = np.dtype(input_dtype).name
    if dtype_name == "uint8":
        return image.astype(np.uint8)
    if dtype_name == "float32":
        return image.astype(np.float32) / 255.0
    if dtype_name == "int8":
        scale, zero_point = input_quantization
        if scale <= 0:
            return np.zeros_like(image, dtype=np.int8)
        normalized = image.astype(np.float32) / 255.0
        quantized = np.rint(normalized / float(scale) + float(zero_point))
        return np.clip(quantized, -128, 127).astype(np.int8)
    raise TFLiteDetectError(f"Unsupported TFLite input dtype: {dtype_name}")


def _drop_batch_dimension(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _looks_like_scores(values: np.ndarray) -> bool:
    if values.ndim != 1:
        return False
    finite = np.isfinite(values)
    if not np.all(finite):
        return False
    return float(np.min(values)) >= -1e-4 and float(np.max(values)) <= 1.0001


def _looks_like_classes(values: np.ndarray) -> bool:
    if values.ndim != 1:
        return False
    finite = np.isfinite(values)
    if not np.all(finite):
        return False
    rounded = np.rint(values)
    mean_abs = float(np.mean(np.abs(values - rounded)))
    return mean_abs <= 1e-3 and float(np.min(rounded)) >= 0.0


def _select_output_tensors(
    outputs: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = [_drop_batch_dimension(np.asarray(value)) for value in outputs]

    boxes_idx = next(
        (idx for idx, arr in enumerate(arrays) if arr.ndim == 2 and arr.shape[1] == 4),
        None,
    )
    if boxes_idx is None:
        raise TFLiteDetectError("Could not locate [N,4] detection boxes output tensor.")

    boxes = arrays[boxes_idx]
    n = int(boxes.shape[0])

    vectors: list[tuple[int, np.ndarray]] = []
    logits: list[tuple[int, np.ndarray]] = []
    for idx, arr in enumerate(arrays):
        if idx == boxes_idx:
            continue
        if arr.ndim == 1 and int(arr.shape[0]) == n:
            vectors.append((idx, arr))
        elif arr.ndim == 2 and int(arr.shape[0]) == n and int(arr.shape[1]) > 1:
            logits.append((idx, arr))

    score_vec = next((arr for _, arr in vectors if _looks_like_scores(arr)), None)
    class_vec = next((arr for _, arr in vectors if _looks_like_classes(arr)), None)

    if score_vec is None and logits:
        score_vec = np.max(logits[0][1], axis=1)
    if class_vec is None and logits:
        class_vec = np.argmax(logits[0][1], axis=1).astype(np.float32)

    if score_vec is None and vectors:
        score_vec = vectors[0][1]
    if class_vec is None:
        for _, arr in vectors:
            if score_vec is not None and arr is score_vec:
                continue
            class_vec = arr
            break

    if score_vec is None or class_vec is None:
        raise TFLiteDetectError("Could not locate class/score output tensors.")

    return boxes, class_vec.astype(np.float32), score_vec.astype(np.float32)


def _to_input_xywh(box: np.ndarray, input_size: float) -> tuple[float, float, float, float]:
    values = [float(v) for v in box[:4]]

    # Common EfficientDet-TFLite contract: [ymin, xmin, ymax, xmax], normalized.
    if max(abs(v) for v in values) <= 1.5:
        y1, x1, y2, x2 = values
        return (
            x1 * input_size,
            y1 * input_size,
            max(0.0, (x2 - x1) * input_size),
            max(0.0, (y2 - y1) * input_size),
        )

    y1, x1, y2, x2 = values
    if y2 > y1 and x2 > x1:
        return (x1, y1, x2 - x1, y2 - y1)

    x, y, w, h = values
    return (x, y, max(0.0, w), max(0.0, h))


def _unletterbox_xywh(
    *,
    box_xywh: tuple[float, float, float, float],
    meta: dict[str, float],
) -> tuple[float, float, float, float] | None:
    x, y, w, h = box_xywh
    scale = meta["scale"]
    pad_x = meta["pad_x"]
    pad_y = meta["pad_y"]
    orig_w = meta["orig_w"]
    orig_h = meta["orig_h"]

    x0 = (x - pad_x) / scale
    y0 = (y - pad_y) / scale
    x1 = (x + w - pad_x) / scale
    y1 = (y + h - pad_y) / scale

    x0 = max(0.0, min(orig_w, x0))
    y0 = max(0.0, min(orig_h, y0))
    x1 = max(0.0, min(orig_w, x1))
    y1 = max(0.0, min(orig_h, y1))

    w0 = x1 - x0
    h0 = y1 - y0
    if w0 <= 0.0 or h0 <= 0.0:
        return None
    return x0, y0, w0, h0


def run_tflite_detection(
    *,
    runtime: TFLiteRuntime,
    image_path: Path,
    score_threshold: float,
    max_detections: int,
) -> tuple[list[TFLiteDetection], dict[str, float]]:
    letterboxed, meta = letterbox_image(
        image_path=image_path,
        input_size=runtime.preprocess.input_size,
    )
    input_tensor = _to_input_tensor(
        image=letterboxed,
        input_dtype=runtime.input_dtype,
        input_quantization=runtime.input_quantization,
    )
    batch = np.expand_dims(input_tensor, axis=0)

    runtime.interpreter.set_tensor(runtime.input_index, batch)
    runtime.interpreter.invoke()

    outputs = [runtime.interpreter.get_tensor(index) for index in runtime.output_indices]
    boxes, classes, scores = _select_output_tensors(outputs)

    ranked_indices = np.argsort(-scores)
    detections: list[TFLiteDetection] = []
    for idx in ranked_indices:
        score = float(scores[idx])
        if score < score_threshold:
            continue

        class_index = int(round(float(classes[idx])))
        input_xywh = _to_input_xywh(boxes[idx], input_size=meta["input_size"])
        original_xywh = _unletterbox_xywh(box_xywh=input_xywh, meta=meta)
        if original_xywh is None:
            continue

        detections.append(
            TFLiteDetection(
                class_index=class_index,
                score=score,
                bbox_xywh=original_xywh,
            )
        )
        if len(detections) >= max_detections:
            break

    return detections, meta
