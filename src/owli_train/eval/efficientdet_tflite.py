from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from owli_train.data.coco import load_coco, validate_coco
from owli_train.tflite_detect import (
    MissingTFLiteDependenciesError,
    create_tflite_runtime,
    ensure_tflite_runtime_dependencies,
    load_tflite_label_map,
    load_tflite_metadata,
    run_tflite_detection,
)


class EfficientDetTFLiteEvalError(RuntimeError):
    """Base class for EfficientDet-TFLite evaluation failures."""


class EfficientDetTFLiteEvalConfigError(EfficientDetTFLiteEvalError):
    """Raised when CLI/config inputs are invalid."""


class MissingEfficientDetTFLiteEvalDependenciesError(EfficientDetTFLiteEvalError):
    """Raised when TensorFlow Lite or pycocotools are unavailable."""


@dataclass(frozen=True)
class EvalEfficientDetTFLiteConfig:
    coco_path: Path
    images_dir: Path
    model_path: Path
    limit_images: int | None
    score_threshold: float
    max_detections_per_image: int
    out_path: Path | None
    category_map_path: Path | None


@dataclass(frozen=True)
class EvalEfficientDetTFLiteArtifacts:
    json_report_path: Path
    markdown_report_path: Path
    model_path: Path


def build_eval_efficientdet_tflite_config(
    *,
    coco_path: Path,
    images_dir: Path,
    model_path: Path,
    limit_images: int | None,
    score_threshold: float,
    max_detections_per_image: int,
    out_path: Path | None,
    category_map_path: Path | None,
) -> EvalEfficientDetTFLiteConfig:
    if Path(model_path).suffix.lower() != ".tflite":
        raise EfficientDetTFLiteEvalConfigError("--model must point to a .tflite file.")
    if limit_images is not None and limit_images <= 0:
        raise EfficientDetTFLiteEvalConfigError("--limit-images must be > 0 when provided.")
    if score_threshold < 0.0 or score_threshold > 1.0:
        raise EfficientDetTFLiteEvalConfigError("--score-threshold must be in [0.0, 1.0].")
    if max_detections_per_image <= 0:
        raise EfficientDetTFLiteEvalConfigError("--max-detections-per-image must be > 0.")

    return EvalEfficientDetTFLiteConfig(
        coco_path=Path(coco_path),
        images_dir=Path(images_dir),
        model_path=Path(model_path),
        limit_images=limit_images,
        score_threshold=score_threshold,
        max_detections_per_image=max_detections_per_image,
        out_path=Path(out_path) if out_path is not None else None,
        category_map_path=Path(category_map_path) if category_map_path is not None else None,
    )


def _ensure_eval_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        tf = ensure_tflite_runtime_dependencies()
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except MissingTFLiteDependenciesError as exc:
        raise MissingEfficientDetTFLiteEvalDependenciesError(
            "EfficientDet TFLite evaluation requires TensorFlow Lite. Install with: "
            "pip install -r requirements\\modelmaker.txt"
        ) from exc
    except Exception as exc:  # pragma: no cover - runtime environment specific
        raise MissingEfficientDetTFLiteEvalDependenciesError(
            "EfficientDet TFLite evaluation requires pycocotools. Install with: "
            "pip install -r requirements\\eval.txt"
        ) from exc

    return tf, COCO, COCOeval, tf


def _default_out_path(cfg: EvalEfficientDetTFLiteConfig) -> Path:
    if cfg.out_path is not None:
        return cfg.out_path

    model_parent = cfg.model_path.parent
    run_dir = model_parent.parent if model_parent.name == "artifacts" else None
    if run_dir is not None:
        return run_dir / "reports" / "eval_efficientdet_tflite.json"

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("work") / "reports" / f"eval-efficientdet-tflite-{stamp}.json"


def _parse_mapping_payload(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    elif suffix == ".json":
        payload = json.loads(text)
    else:
        try:
            payload = json.loads(text)
        except Exception:
            payload = yaml.safe_load(text)

    if not isinstance(payload, dict):
        raise EfficientDetTFLiteEvalConfigError("--category-map must be a JSON/YAML object.")
    return payload


def _load_category_mapping(
    *,
    eval_coco: dict[str, Any],
    label_map: list[str],
    category_map_path: Path | None,
) -> tuple[dict[int, int], str]:
    categories = eval_coco.get("categories") or []
    by_id = {int(item["id"]): item for item in categories}
    by_name = {str(item["name"]): item for item in categories}

    if category_map_path is not None:
        payload = _parse_mapping_payload(category_map_path)
        mapping: dict[int, int] = {}
        for key, value in payload.items():
            if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
                class_index = int(key)
            elif isinstance(key, str) and key in label_map:
                class_index = label_map.index(key)
            else:
                raise EfficientDetTFLiteEvalConfigError(
                    f"Invalid category-map key: {key}. Use class index or known class name."
                )

            if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
                category_id = int(value)
                if category_id not in by_id:
                    raise EfficientDetTFLiteEvalConfigError(
                        f"Category-map references unknown category id: {category_id}"
                    )
            elif isinstance(value, str) and value in by_name:
                category_id = int(by_name[value]["id"])
            else:
                raise EfficientDetTFLiteEvalConfigError(
                    f"Invalid category-map value: {value}. Use category id or known category name."
                )
            mapping[class_index] = category_id
        return mapping, "explicit_map"

    if label_map:
        mapping: dict[int, int] = {}
        missing: list[str] = []
        for class_index, class_name in enumerate(label_map):
            if class_name in by_name:
                mapping[class_index] = int(by_name[class_name]["id"])
            else:
                missing.append(class_name)
        if mapping:
            if missing:
                raise EfficientDetTFLiteEvalConfigError(
                    "Could not align class names to COCO categories: "
                    f"{', '.join(missing)}. Provide --category-map to override."
                )
            return mapping, "label_name_match"

    sorted_ids = sorted(by_id)
    mapping = {idx: category_id for idx, category_id in enumerate(sorted_ids)}
    return mapping, "contiguous_by_category_id"


def _iou_xywh(
    box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]
) -> float:
    ax0, ay0, aw, ah = box_a
    bx0, by0, bw, bh = box_b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)

    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def _compute_per_class_counts(
    *,
    eval_images: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    score_threshold: float,
    categories: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    gt_by_image_category: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}
    for ann in annotations:
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        bbox = ann.get("bbox") or []
        if len(bbox) != 4:
            continue
        gt_by_image_category.setdefault((image_id, category_id), []).append(
            (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        )

    pred_by_image_category: dict[
        tuple[int, int], list[tuple[float, tuple[float, float, float, float]]]
    ] = {}
    for det in detections:
        score = float(det["score"])
        if score < score_threshold:
            continue
        image_id = int(det["image_id"])
        category_id = int(det["category_id"])
        bbox = det.get("bbox") or []
        if len(bbox) != 4:
            continue
        pred_by_image_category.setdefault((image_id, category_id), []).append(
            (score, (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
        )

    category_ids = [int(item["id"]) for item in categories]
    by_category: dict[int, dict[str, int]] = {
        category_id: {"tp": 0, "fp": 0, "fn": 0} for category_id in category_ids
    }

    eval_image_ids = [int(item["id"]) for item in eval_images]
    for image_id in eval_image_ids:
        for category_id in category_ids:
            gt_boxes = gt_by_image_category.get((image_id, category_id), [])
            preds = pred_by_image_category.get((image_id, category_id), [])
            preds = sorted(preds, key=lambda item: -item[0])

            matched = set()
            tp = 0
            fp = 0
            for _, pred_box in preds:
                best_iou = 0.0
                best_index = -1
                for idx, gt_box in enumerate(gt_boxes):
                    if idx in matched:
                        continue
                    iou = _iou_xywh(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_index = idx
                if best_index >= 0 and best_iou >= 0.5:
                    matched.add(best_index)
                    tp += 1
                else:
                    fp += 1

            fn = max(0, len(gt_boxes) - len(matched))
            by_category[category_id]["tp"] += tp
            by_category[category_id]["fp"] += fp
            by_category[category_id]["fn"] += fn

    category_names = {int(item["id"]): str(item["name"]) for item in categories}
    per_class: dict[str, dict[str, Any]] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for category_id in sorted(category_ids):
        counts = by_category[category_id]
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        per_class[category_names[category_id]] = {
            "category_id": category_id,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "predictions": tp + fp,
            "ground_truth": tp + fn,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    totals = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0,
        "recall": float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0,
    }
    return per_class, totals


def _metrics_from_stats(stats: Any) -> dict[str, float]:
    names = [
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
    return {name: float(stats[idx]) for idx, name in enumerate(names)}


def _zero_metrics() -> dict[str, float]:
    return {
        "AP": 0.0,
        "AP50": 0.0,
        "AP75": 0.0,
        "AP_small": 0.0,
        "AP_medium": 0.0,
        "AP_large": 0.0,
        "AR1": 0.0,
        "AR10": 0.0,
        "AR100": 0.0,
        "AR_small": 0.0,
        "AR_medium": 0.0,
        "AR_large": 0.0,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    noise = report["noise_metric"]
    counts = report["summary_counts"]
    lines = [
        "# EfficientDet TFLite Evaluation Summary",
        "",
        f"- Model: `{report['model_path']}`",
        f"- COCO: `{report['coco_path']}`",
        f"- Images dir: `{report['images_dir']}`",
        f"- Evaluated images: `{report['num_eval_images']}`",
        f"- Detections kept: `{report['num_detections']}`",
        "",
        "## COCO Metrics",
        "",
        f"- AP: `{metrics['AP']:.4f}`",
        f"- AP50: `{metrics['AP50']:.4f}`",
        f"- AP75: `{metrics['AP75']:.4f}`",
        f"- AR100: `{metrics['AR100']:.4f}`",
        "",
        "## Noise Metric",
        "",
        f"- Score threshold: `{noise['score_threshold']:.3f}`",
        f"- False positives: `{noise['false_positives']}`",
        f"- FP per 100 images: `{noise['fp_per_100_images']:.3f}`",
        "",
        "## Per-Class Aggregate",
        "",
        f"- TP: `{counts['tp']}`",
        f"- FP: `{counts['fp']}`",
        f"- FN: `{counts['fn']}`",
        f"- Precision: `{counts['precision']:.4f}`",
        f"- Recall: `{counts['recall']:.4f}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_efficientdet_tflite(
    cfg: EvalEfficientDetTFLiteConfig,
) -> EvalEfficientDetTFLiteArtifacts:
    tf, COCO, COCOeval, _ = _ensure_eval_dependencies()

    eval_coco = load_coco(cfg.coco_path)
    validate_coco(eval_coco, images_dir=cfg.images_dir)

    output_json = _default_out_path(cfg)
    output_md = output_json.with_suffix(".md")
    metadata = load_tflite_metadata(cfg.model_path)
    label_map = load_tflite_label_map(cfg.model_path, metadata)

    class_to_category, mapping_source = _load_category_mapping(
        eval_coco=eval_coco,
        label_map=label_map.class_names,
        category_map_path=cfg.category_map_path,
    )

    runtime = create_tflite_runtime(model_path=cfg.model_path, tf=tf)

    eval_images = sorted(eval_coco["images"], key=lambda item: int(item["id"]))
    if cfg.limit_images is not None:
        eval_images = eval_images[: cfg.limit_images]
    eval_image_ids = {int(item["id"]) for item in eval_images}

    detections: list[dict[str, Any]] = []
    for image in eval_images:
        image_id = int(image["id"])
        image_path = cfg.images_dir / str(image["file_name"])
        predicted, _ = run_tflite_detection(
            runtime=runtime,
            image_path=image_path,
            score_threshold=cfg.score_threshold,
            max_detections=cfg.max_detections_per_image,
        )

        for item in predicted:
            if item.class_index not in class_to_category:
                continue
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": class_to_category[item.class_index],
                    "bbox": [float(v) for v in item.bbox_xywh],
                    "score": float(item.score),
                }
            )

    if detections:
        coco_gt = COCO(str(cfg.coco_path))
        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = sorted(eval_image_ids)
        coco_eval.params.maxDets = [1, 10, cfg.max_detections_per_image]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        metrics = _metrics_from_stats(coco_eval.stats)
    else:
        metrics = _zero_metrics()

    eval_annotations = [
        ann
        for ann in (eval_coco.get("annotations") or [])
        if int(ann.get("image_id", -1)) in eval_image_ids
    ]
    per_class, totals = _compute_per_class_counts(
        eval_images=eval_images,
        detections=detections,
        annotations=eval_annotations,
        score_threshold=cfg.score_threshold,
        categories=eval_coco["categories"],
    )

    num_images = len(eval_images)
    fp_per_100 = float(totals["fp"] * 100.0 / num_images) if num_images > 0 else 0.0
    noise_metric = {
        "score_threshold": cfg.score_threshold,
        "false_positives": totals["fp"],
        "fp_per_100_images": fp_per_100,
    }

    report = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model_path": str(cfg.model_path),
        "coco_path": str(cfg.coco_path),
        "images_dir": str(cfg.images_dir),
        "limit_images": cfg.limit_images,
        "score_threshold": cfg.score_threshold,
        "max_detections_per_image": cfg.max_detections_per_image,
        "num_eval_images": num_images,
        "num_detections": len(detections),
        "metrics": metrics,
        "summary_counts": totals,
        "per_class": per_class,
        "noise_metric": noise_metric,
        "mapping": {
            "source": mapping_source,
            "label_source": label_map.source,
            "class_count": len(label_map.class_names),
        },
        "tflite_io": {
            "builtin_ops_only": runtime.builtin_ops_only,
            "operator_names": runtime.operator_names,
            "inputs": runtime.inspect_inputs,
            "outputs": runtime.inspect_outputs,
            "preprocess": {
                "input_size": runtime.preprocess.input_size,
                "input_shape": runtime.preprocess.input_shape,
                "input_dtype": runtime.preprocess.input_dtype,
                "normalization": runtime.preprocess.normalization,
                "resize_policy": runtime.preprocess.resize_policy,
                "color_space": runtime.preprocess.color_space,
            },
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(output_md, report)
    return EvalEfficientDetTFLiteArtifacts(
        json_report_path=output_json,
        markdown_report_path=output_md,
        model_path=cfg.model_path,
    )
