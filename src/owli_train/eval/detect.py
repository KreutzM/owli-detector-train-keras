from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from owli_train.data.coco import load_coco, validate_coco


class EvalError(RuntimeError):
    """Base class for detector evaluation failures."""


class EvalConfigError(EvalError):
    """Raised when CLI/config inputs for evaluation are invalid."""


class MissingEvalDependenciesError(EvalError):
    """Raised when TensorFlow or pycocotools are missing."""


@dataclass(frozen=True)
class EvalDetectConfig:
    coco_path: Path
    images_dir: Path
    run_dir: Path | None
    model_path: Path | None
    limit_images: int | None
    score_threshold: float
    max_detections_per_image: int
    out_path: Path | None
    category_map_path: Path | None


@dataclass(frozen=True)
class EvalArtifacts:
    json_report_path: Path
    markdown_report_path: Path
    model_path: Path
    run_dir: Path | None


def build_eval_detect_config(
    *,
    coco_path: Path,
    images_dir: Path,
    run_dir: Path | None,
    model_path: Path | None,
    limit_images: int | None,
    score_threshold: float,
    max_detections_per_image: int,
    out_path: Path | None,
    category_map_path: Path | None,
) -> EvalDetectConfig:
    if run_dir is None and model_path is None:
        raise EvalConfigError("Provide either --run-dir or --model.")
    if run_dir is not None and model_path is not None:
        raise EvalConfigError("Use either --run-dir or --model, not both.")
    if limit_images is not None and limit_images <= 0:
        raise EvalConfigError("--limit-images must be > 0 when provided.")
    if score_threshold < 0.0 or score_threshold > 1.0:
        raise EvalConfigError("--score-threshold must be in [0.0, 1.0].")
    if max_detections_per_image <= 0:
        raise EvalConfigError("--max-detections-per-image must be > 0.")

    return EvalDetectConfig(
        coco_path=Path(coco_path),
        images_dir=Path(images_dir),
        run_dir=Path(run_dir) if run_dir is not None else None,
        model_path=Path(model_path) if model_path is not None else None,
        limit_images=limit_images,
        score_threshold=score_threshold,
        max_detections_per_image=max_detections_per_image,
        out_path=Path(out_path) if out_path is not None else None,
        category_map_path=Path(category_map_path) if category_map_path is not None else None,
    )


def ensure_eval_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import tensorflow as tf
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:  # pragma: no cover - runtime environment specific
        raise MissingEvalDependenciesError(
            "Evaluation requires TensorFlow and pycocotools. Install with: "
            "pip install -r requirements\\keras.txt and pip install -r requirements\\eval.txt"
        ) from exc

    return tf, COCO, COCOeval, tf.keras.models


def _resolve_model_source(cfg: EvalDetectConfig) -> Path:
    if cfg.model_path is not None:
        return cfg.model_path

    assert cfg.run_dir is not None
    keras_path = cfg.run_dir / "artifacts" / "detector.keras"
    saved_model_path = cfg.run_dir / "artifacts" / "saved_model"
    if keras_path.is_file():
        return keras_path
    if saved_model_path.exists():
        return saved_model_path

    raise EvalConfigError(
        "Could not resolve model artifact from run dir. Expected one of: "
        f"{keras_path} or {saved_model_path}"
    )


def _default_out_path(cfg: EvalDetectConfig) -> Path:
    if cfg.out_path is not None:
        return cfg.out_path

    if cfg.run_dir is not None:
        return cfg.run_dir / "reports" / "eval.json"

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("work") / "reports" / f"eval-{stamp}.json"


def _load_snapshot_class_names(run_dir: Path | None) -> tuple[list[str] | None, str | None]:
    if run_dir is None:
        return None, None

    snapshot_path = run_dir / "label_map_snapshot.json"
    if not snapshot_path.is_file():
        return None, None

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    class_names = payload.get("class_names") or []
    bbox_format = payload.get("bounding_box_format")
    if not isinstance(class_names, list):
        return None, bbox_format if isinstance(bbox_format, str) else None

    names = [str(v) for v in class_names]
    fmt = bbox_format if isinstance(bbox_format, str) else None
    return names, fmt


def _load_category_map_file(
    path: Path,
    eval_categories_by_id: dict[int, dict[str, Any]],
    eval_categories_by_name: dict[str, dict[str, Any]],
    class_names: list[str] | None,
) -> dict[int, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise EvalConfigError("--category-map must be a JSON object.")

    mapping: dict[int, int] = {}
    for raw_key, raw_value in raw.items():
        class_index: int
        if isinstance(raw_key, int) or (isinstance(raw_key, str) and raw_key.isdigit()):
            class_index = int(raw_key)
        elif isinstance(raw_key, str) and class_names and raw_key in class_names:
            class_index = class_names.index(raw_key)
        else:
            raise EvalConfigError(
                f"Invalid category-map key: {raw_key}. Use class index or known class name."
            )

        if isinstance(raw_value, int) or (isinstance(raw_value, str) and raw_value.isdigit()):
            cat_id = int(raw_value)
            if cat_id not in eval_categories_by_id:
                raise EvalConfigError(f"Category-map references unknown category id: {cat_id}")
        elif isinstance(raw_value, str) and raw_value in eval_categories_by_name:
            cat_id = int(eval_categories_by_name[raw_value]["id"])
        else:
            raise EvalConfigError(
                f"Invalid category-map value: {raw_value}. Use category id or known category name."
            )

        mapping[class_index] = cat_id

    return mapping


def _build_class_to_category_mapping(
    eval_coco: dict[str, Any],
    class_names: list[str] | None,
    category_map_path: Path | None,
) -> dict[int, int]:
    categories = eval_coco["categories"]
    categories_by_id = {int(c["id"]): c for c in categories}
    categories_by_name = {str(c["name"]): c for c in categories}

    if category_map_path is not None:
        return _load_category_map_file(
            path=category_map_path,
            eval_categories_by_id=categories_by_id,
            eval_categories_by_name=categories_by_name,
            class_names=class_names,
        )

    if class_names:
        mapping: dict[int, int] = {}
        missing: list[str] = []
        for idx, class_name in enumerate(class_names):
            if class_name not in categories_by_name:
                missing.append(class_name)
                continue
            mapping[idx] = int(categories_by_name[class_name]["id"])

        if missing:
            raise EvalConfigError(
                "Eval category alignment failed for class_names: "
                f"{', '.join(missing)}. Provide --category-map to override."
            )
        return mapping

    sorted_ids = sorted(categories_by_id)
    return {idx: cat_id for idx, cat_id in enumerate(sorted_ids)}


def _infer_input_size(model: Any) -> int:
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, tuple) and len(shape) >= 3:
        h = shape[1]
        w = shape[2]
        if isinstance(h, int) and h > 0 and isinstance(w, int) and w > 0:
            return max(h, w)
    return 640


def _letterbox_image(image_path: Path, input_size: int) -> tuple[np.ndarray, dict[str, float]]:
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

    arr = np.asarray(canvas, dtype=np.float32)
    meta = {
        "orig_w": float(orig_w),
        "orig_h": float(orig_h),
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "input_size": float(input_size),
    }
    return arr, meta


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _extract_model_outputs(prediction: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(prediction, dict):
        raise EvalError("Unsupported prediction output type. Expected dict-like output.")

    if "boxes" in prediction:
        boxes = _to_numpy(prediction["boxes"])
    elif "bbox" in prediction:
        boxes = _to_numpy(prediction["bbox"])
    else:
        raise EvalError("Prediction output missing 'boxes'.")

    if "classes" in prediction:
        classes = _to_numpy(prediction["classes"])
    else:
        raise EvalError("Prediction output missing 'classes'.")

    if "confidence" in prediction:
        scores = _to_numpy(prediction["confidence"])
    elif "scores" in prediction:
        scores = _to_numpy(prediction["scores"])
    else:
        raise EvalError("Prediction output missing 'confidence' or 'scores'.")

    if boxes.ndim == 3:
        boxes = boxes[0]
    if classes.ndim == 2:
        classes = classes[0]
    if scores.ndim == 2:
        scores = scores[0]

    return boxes, classes, scores


def _to_xywh(box: np.ndarray, fmt: str, input_size: float) -> tuple[float, float, float, float]:
    x1: float
    y1: float
    x2: float
    y2: float

    if fmt == "xywh":
        x1 = float(box[0])
        y1 = float(box[1])
        w = float(box[2])
        h = float(box[3])
        return x1, y1, w, h

    if fmt == "xyxy":
        x1 = float(box[0])
        y1 = float(box[1])
        x2 = float(box[2])
        y2 = float(box[3])
        return x1, y1, x2 - x1, y2 - y1

    if fmt == "rel_xywh":
        x1 = float(box[0]) * input_size
        y1 = float(box[1]) * input_size
        w = float(box[2]) * input_size
        h = float(box[3]) * input_size
        return x1, y1, w, h

    if fmt == "rel_xyxy":
        x1 = float(box[0]) * input_size
        y1 = float(box[1]) * input_size
        x2 = float(box[2]) * input_size
        y2 = float(box[3]) * input_size
        return x1, y1, x2 - x1, y2 - y1

    raise EvalError(f"Unsupported bounding box format: {fmt}")


def _unletterbox_xywh(
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


def _run_inference(
    *,
    tf: Any,
    model: Any,
    eval_images: list[dict[str, Any]],
    images_dir: Path,
    class_to_category: dict[int, int],
    bbox_format: str,
    score_threshold: float,
    max_detections_per_image: int,
) -> list[dict[str, Any]]:
    input_size = _infer_input_size(model)
    detections: list[dict[str, Any]] = []

    for image in eval_images:
        image_id = int(image["id"])
        image_path = images_dir / str(image["file_name"])

        arr, meta = _letterbox_image(image_path, input_size=input_size)
        batch = np.expand_dims(arr, axis=0)

        prediction = model(tf.convert_to_tensor(batch), training=False)
        boxes, classes, scores = _extract_model_outputs(prediction)

        ranking = np.argsort(-scores)
        kept = 0
        for idx in ranking:
            score = float(scores[idx])
            if score < score_threshold:
                continue

            class_idx = int(round(float(classes[idx])))
            if class_idx not in class_to_category:
                continue

            xywh_model = _to_xywh(
                boxes[idx],
                fmt=bbox_format,
                input_size=meta["input_size"],
            )
            xywh = _unletterbox_xywh(xywh_model, meta)
            if xywh is None:
                continue

            detections.append(
                {
                    "image_id": image_id,
                    "category_id": class_to_category[class_idx],
                    "bbox": [float(v) for v in xywh],
                    "score": score,
                }
            )

            kept += 1
            if kept >= max_detections_per_image:
                break

    return detections


def _metrics_from_stats(stats: np.ndarray) -> dict[str, float]:
    keys = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    return {key: float(stats[idx]) for idx, key in enumerate(keys)}


def _write_markdown_summary(path: Path, report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    lines = [
        "# Detection Evaluation Summary",
        "",
        f"- Run dir: `{report['run_dir']}`",
        f"- Model: `{report['model_path']}`",
        f"- COCO: `{report['coco_path']}`",
        f"- Images dir: `{report['images_dir']}`",
        f"- Evaluated images: `{report['num_eval_images']}`",
        f"- Detections kept: `{report['num_detections']}`",
        "",
        "## Key Metrics",
        "",
        f"- AP: `{metrics['AP']:.4f}`",
        f"- AP50: `{metrics['AP50']:.4f}`",
        f"- AP75: `{metrics['AP75']:.4f}`",
        f"- AR100: `{metrics['AR100']:.4f}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_detect(cfg: EvalDetectConfig) -> EvalArtifacts:
    tf, COCO, COCOeval, keras_models = ensure_eval_dependencies()

    eval_coco = load_coco(cfg.coco_path)
    validate_coco(eval_coco, images_dir=cfg.images_dir)

    model_path = _resolve_model_source(cfg)
    output_json = _default_out_path(cfg)
    output_md = output_json.with_suffix(".md")

    class_names, snapshot_bbox_format = _load_snapshot_class_names(cfg.run_dir)
    class_to_category = _build_class_to_category_mapping(
        eval_coco=eval_coco,
        class_names=class_names,
        category_map_path=cfg.category_map_path,
    )

    model = keras_models.load_model(str(model_path), compile=False)

    model_bbox_format = snapshot_bbox_format or getattr(model, "bounding_box_format", "xywh")
    if not isinstance(model_bbox_format, str):
        model_bbox_format = "xywh"

    eval_images = sorted(eval_coco["images"], key=lambda item: int(item["id"]))
    if cfg.limit_images is not None:
        eval_images = eval_images[: cfg.limit_images]

    detections = _run_inference(
        tf=tf,
        model=model,
        eval_images=eval_images,
        images_dir=cfg.images_dir,
        class_to_category=class_to_category,
        bbox_format=model_bbox_format,
        score_threshold=cfg.score_threshold,
        max_detections_per_image=cfg.max_detections_per_image,
    )

    coco_gt = COCO(str(cfg.coco_path))
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = [int(img["id"]) for img in eval_images]
    coco_eval.params.maxDets = [1, 10, cfg.max_detections_per_image]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = _metrics_from_stats(coco_eval.stats)

    report = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run_dir": str(cfg.run_dir) if cfg.run_dir is not None else None,
        "model_path": str(model_path),
        "coco_path": str(cfg.coco_path),
        "images_dir": str(cfg.images_dir),
        "bbox_format": model_bbox_format,
        "score_threshold": cfg.score_threshold,
        "max_detections_per_image": cfg.max_detections_per_image,
        "limit_images": cfg.limit_images,
        "num_eval_images": len(eval_images),
        "num_detections": len(detections),
        "metrics": metrics,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown_summary(output_md, report)

    return EvalArtifacts(
        json_report_path=output_json,
        markdown_report_path=output_md,
        model_path=model_path,
        run_dir=cfg.run_dir,
    )
