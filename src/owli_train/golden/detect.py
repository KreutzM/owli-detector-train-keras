from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from owli_train.tflite_detect import (
    MissingTFLiteDependenciesError,
    TFLiteDetection,
    create_tflite_runtime,
    ensure_tflite_runtime_dependencies,
    load_tflite_label_map,
    load_tflite_metadata,
    run_tflite_detection,
)


class GoldenDetectError(RuntimeError):
    """Base class for golden sample generation failures."""


class GoldenDetectConfigError(GoldenDetectError):
    """Raised when CLI/config inputs are invalid."""


class MissingGoldenDependenciesError(GoldenDetectError):
    """Raised when TensorFlow Lite runtime is unavailable."""


@dataclass(frozen=True)
class GoldenDetectConfig:
    model_path: Path
    image_path: Path
    out_path: Path
    score_threshold: float
    max_results: int
    num_threads: int | None


@dataclass(frozen=True)
class GoldenDetectArtifacts:
    out_path: Path
    num_detections: int


def build_golden_detect_config(
    *,
    model_path: Path,
    image_path: Path,
    out_path: Path,
    score_threshold: float,
    max_results: int,
    num_threads: int | None = None,
) -> GoldenDetectConfig:
    if Path(model_path).suffix.lower() != ".tflite":
        raise GoldenDetectConfigError("--model must point to a .tflite file.")
    if score_threshold < 0.0 or score_threshold > 1.0:
        raise GoldenDetectConfigError("--score-threshold must be in [0.0, 1.0].")
    if max_results <= 0:
        raise GoldenDetectConfigError("--max-results must be > 0.")
    if num_threads is not None and int(num_threads) <= 0:
        raise GoldenDetectConfigError("--num-threads must be > 0 when provided.")

    return GoldenDetectConfig(
        model_path=Path(model_path),
        image_path=Path(image_path),
        out_path=Path(out_path),
        score_threshold=score_threshold,
        max_results=max_results,
        num_threads=int(num_threads) if num_threads is not None else None,
    )


def _detection_to_payload(
    *,
    item: TFLiteDetection,
    class_names: list[str],
) -> dict[str, Any]:
    class_index = int(item.class_index)
    if 0 <= class_index < len(class_names):
        class_name = class_names[class_index]
    else:
        class_name = f"class_{class_index}"

    return {
        "class_index": class_index,
        "class_name": class_name,
        "score": float(item.score),
        "bbox": [float(v) for v in item.bbox_xywh],
        "bbox_format": "xywh",
        "coordinates": "absolute_pixels",
    }


def build_golden_payload(
    *,
    cfg: GoldenDetectConfig,
    preprocess: dict[str, Any],
    detections: list[TFLiteDetection],
    class_names: list[str],
    class_source: str,
    model_metadata: dict[str, Any] | None,
    inspect_summary: dict[str, Any],
) -> dict[str, Any]:
    detection_payload = [
        _detection_to_payload(item=item, class_names=class_names)
        for item in detections[: cfg.max_results]
    ]
    return {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model_path": str(cfg.model_path),
        "image_path": str(cfg.image_path),
        "contract": {
            "input_preprocessing": preprocess,
            "bbox_format": "xywh",
            "coordinates": "absolute_pixels",
            "class_labels_source": class_source,
            "score_threshold": cfg.score_threshold,
            "max_results": cfg.max_results,
        },
        "model_metadata": model_metadata,
        "inspect_tflite": inspect_summary,
        "detections": detection_payload,
    }


def generate_golden_detect(cfg: GoldenDetectConfig) -> GoldenDetectArtifacts:
    try:
        tf = ensure_tflite_runtime_dependencies()
    except MissingTFLiteDependenciesError as exc:
        raise MissingGoldenDependenciesError(
            "Golden sample generation requires TensorFlow Lite runtime. Install with: "
            "pip install -r requirements\\modelmaker.txt"
        ) from exc

    runtime = create_tflite_runtime(model_path=cfg.model_path, tf=tf, num_threads=cfg.num_threads)
    model_metadata = load_tflite_metadata(cfg.model_path)
    labels = load_tflite_label_map(cfg.model_path, model_metadata)
    detections, _ = run_tflite_detection(
        runtime=runtime,
        image_path=cfg.image_path,
        score_threshold=cfg.score_threshold,
        max_detections=cfg.max_results,
    )

    preprocess = {
        "resize_policy": runtime.preprocess.resize_policy,
        "target_size": runtime.preprocess.input_size,
        "input_shape": runtime.preprocess.input_shape,
        "input_dtype": runtime.preprocess.input_dtype,
        "normalization": runtime.preprocess.normalization,
        "color_space": runtime.preprocess.color_space,
        "pad_value": runtime.preprocess.pad_value,
    }
    inspect_summary = {
        "builtin_ops_only": runtime.builtin_ops_only,
        "operator_names": runtime.operator_names,
        "inputs": runtime.inspect_inputs,
        "outputs": runtime.inspect_outputs,
    }
    payload = build_golden_payload(
        cfg=cfg,
        preprocess=preprocess,
        detections=detections,
        class_names=labels.class_names,
        class_source=labels.source,
        model_metadata=model_metadata,
        inspect_summary=inspect_summary,
    )

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return GoldenDetectArtifacts(out_path=cfg.out_path, num_detections=len(payload["detections"]))
