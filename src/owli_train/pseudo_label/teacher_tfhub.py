from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from owli_train.pseudo_label.coco_writer import (
    build_pseudo_coco,
    build_pseudo_report,
    load_coco80_categories,
    parse_classes_filter,
    write_json,
)

DEFAULT_TFHUB_TEACHER = "https://tfhub.dev/tensorflow/efficientdet/d2/1"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class TeacherPseudoLabelError(RuntimeError):
    """Base class for pseudo-labeling failures."""


class TeacherPseudoLabelConfigError(TeacherPseudoLabelError):
    """Raised when pseudo-label CLI/config parameters are invalid."""


class MissingTeacherDependenciesError(TeacherPseudoLabelError):
    """Raised when TensorFlow/TF Hub dependencies are missing."""


@dataclass(frozen=True)
class TeacherPseudoLabelConfig:
    images_dir: Path
    out_path: Path
    report_out_path: Path
    teacher: str
    teacher_savedmodel: Path | None
    batch_size: int
    input_size: int
    score_threshold: float
    max_detections_per_image: int
    classes_filter: str | None
    limit_images: int | None
    seed: int
    num_parallel_calls: int
    prefetch_buffer: int
    debug_io: bool
    viz_out_dir: Path | None
    viz_max_images: int


@dataclass(frozen=True)
class TeacherPseudoLabelArtifacts:
    pseudo_coco_path: Path
    report_path: Path
    images_processed: int
    detections_kept: int
    teacher_source: str
    elapsed_seconds: float


@dataclass(frozen=True)
class TeacherModel:
    runner: Any
    source: str
    input_dtype_name: str
    signature_name: str


def _ensure_teacher_dependencies() -> tuple[Any, Any]:
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - runtime environment specific
        raise MissingTeacherDependenciesError(
            "Teacher pseudo-labeling requires TensorFlow + TF Hub. Install with: "
            "pip install -r requirements\\teacher.txt"
        ) from exc
    try:
        import tensorflow_hub as hub
    except Exception as exc:  # pragma: no cover - runtime environment specific
        raise MissingTeacherDependenciesError(
            "Teacher pseudo-labeling requires tensorflow-hub. Install with: "
            "pip install -r requirements\\teacher.txt"
        ) from exc
    return tf, hub


def _default_report_path(out_path: Path) -> Path:
    return out_path.with_suffix(".report.json")


def build_teacher_pseudo_label_config(
    *,
    images_dir: Path,
    out_path: Path,
    teacher: str,
    teacher_savedmodel: Path | None,
    batch_size: int,
    input_size: int,
    score_threshold: float,
    max_detections_per_image: int,
    classes_filter: str | None,
    limit_images: int | None,
    seed: int,
    num_parallel_calls: int,
    prefetch_buffer: int,
    debug_io: bool,
    report_out_path: Path | None,
    viz_out_dir: Path | None,
    viz_max_images: int,
) -> TeacherPseudoLabelConfig:
    images_path = Path(images_dir)
    if not images_path.is_dir():
        raise TeacherPseudoLabelConfigError(f"--images-dir is not a directory: {images_path}")

    out = Path(out_path)
    if out.suffix.lower() != ".json":
        raise TeacherPseudoLabelConfigError("--out must be a .json path.")

    savedmodel_path = Path(teacher_savedmodel) if teacher_savedmodel is not None else None
    if savedmodel_path is not None and not savedmodel_path.is_dir():
        raise TeacherPseudoLabelConfigError(
            f"--teacher-savedmodel must point to a directory: {savedmodel_path}"
        )
    if batch_size <= 0:
        raise TeacherPseudoLabelConfigError("--batch-size must be > 0.")
    if input_size <= 0:
        raise TeacherPseudoLabelConfigError("--input-size must be > 0.")
    if score_threshold < 0.0 or score_threshold > 1.0:
        raise TeacherPseudoLabelConfigError("--score-threshold must be in [0.0, 1.0].")
    if max_detections_per_image <= 0:
        raise TeacherPseudoLabelConfigError("--max-detections-per-image must be > 0.")
    if limit_images is not None and limit_images <= 0:
        raise TeacherPseudoLabelConfigError("--limit-images must be > 0 when provided.")
    if num_parallel_calls <= 0:
        raise TeacherPseudoLabelConfigError("--num-parallel-calls must be > 0.")
    if prefetch_buffer <= 0:
        raise TeacherPseudoLabelConfigError("--prefetch-buffer must be > 0.")
    if viz_max_images <= 0:
        raise TeacherPseudoLabelConfigError("--viz-max-images must be > 0.")

    report = Path(report_out_path) if report_out_path is not None else _default_report_path(out)
    return TeacherPseudoLabelConfig(
        images_dir=images_path,
        out_path=out,
        report_out_path=report,
        teacher=teacher,
        teacher_savedmodel=savedmodel_path,
        batch_size=batch_size,
        input_size=input_size,
        score_threshold=score_threshold,
        max_detections_per_image=max_detections_per_image,
        classes_filter=classes_filter,
        limit_images=limit_images,
        seed=int(seed),
        num_parallel_calls=int(num_parallel_calls),
        prefetch_buffer=int(prefetch_buffer),
        debug_io=bool(debug_io),
        viz_out_dir=Path(viz_out_dir) if viz_out_dir is not None else None,
        viz_max_images=int(viz_max_images),
    )


def _find_input_dtype_name(signature: Any) -> str:
    try:
        _, kwargs = signature.structured_input_signature
    except Exception:
        return "uint8"
    for spec in kwargs.values():
        dtype_name = getattr(getattr(spec, "dtype", None), "name", "")
        if dtype_name:
            return str(dtype_name)
    return "uint8"


def load_teacher(model_handle_or_path: str) -> TeacherModel:
    tf, hub = _ensure_teacher_dependencies()
    model_path = Path(model_handle_or_path)
    source = model_handle_or_path

    if model_path.is_dir():
        model = tf.saved_model.load(str(model_path))
        source = str(model_path)
    else:
        model = hub.load(model_handle_or_path)

    signature_name = "callable"
    runner: Any
    signatures = model.signatures if hasattr(model, "signatures") else None
    if signatures:
        if "serving_default" in signatures:
            signature_name = "serving_default"
            runner = signatures["serving_default"]
        else:
            signature_name, runner = next(iter(signatures.items()))
    elif callable(model):
        runner = model
    else:  # pragma: no cover - runtime environment specific
        raise TeacherPseudoLabelError("Could not resolve teacher callable from model handle/path.")

    return TeacherModel(
        runner=runner,
        source=source,
        input_dtype_name=_find_input_dtype_name(runner),
        signature_name=signature_name,
    )


def _discover_images(images_dir: Path) -> list[Path]:
    paths = [
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTS
    ]
    return paths


def _build_tf_dataset(
    *,
    tf: Any,
    image_paths: list[Path],
    rel_names: list[str],
    input_size: int,
    batch_size: int,
    num_parallel_calls: int,
    prefetch_buffer: int,
    input_dtype_name: str,
) -> Any:
    abs_paths = [str(path) for path in image_paths]
    ds = tf.data.Dataset.from_tensor_slices((abs_paths, rel_names))

    def _decode_and_resize(path: Any, rel_name: Any) -> dict[str, Any]:
        data = tf.io.read_file(path)
        image = tf.image.decode_image(data, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        orig_h = tf.shape(image)[0]
        orig_w = tf.shape(image)[1]
        resized = tf.image.resize_with_pad(tf.cast(image, tf.float32), input_size, input_size)
        if input_dtype_name == "float32":
            model_input = resized / 255.0
        else:
            clipped = tf.clip_by_value(tf.round(resized), 0.0, 255.0)
            model_input = tf.cast(clipped, tf.uint8)
        return {
            "image": model_input,
            "file_name": rel_name,
            "orig_h": tf.cast(orig_h, tf.int32),
            "orig_w": tf.cast(orig_w, tf.int32),
        }

    ds = ds.map(_decode_and_resize, num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(prefetch_buffer)
    return ds


def _resolve_output(outputs: Any) -> dict[str, Any]:
    if isinstance(outputs, dict):
        return outputs
    if hasattr(outputs, "items"):
        return dict(outputs.items())
    if hasattr(outputs, "_asdict"):
        return outputs._asdict()
    raise TeacherPseudoLabelError("Teacher outputs are not a dict-like object.")


def _extract_detection_arrays(
    output_dict: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def _pick(*names: str) -> Any | None:
        for name in names:
            if name in output_dict:
                return output_dict[name]
        return None

    boxes = _pick("detection_boxes", "boxes", "output_0")
    scores = _pick("detection_scores", "scores", "output_1")
    classes = _pick("detection_classes", "classes", "output_2")
    num = _pick("num_detections", "valid_detections", "output_3")
    if boxes is None or scores is None or classes is None:
        keys = ", ".join(sorted(output_dict.keys()))
        raise TeacherPseudoLabelError(
            f"Could not find detection outputs in teacher result. Available keys: {keys}"
        )

    boxes_np = np.asarray(boxes)
    scores_np = np.asarray(scores)
    classes_np = np.asarray(classes)
    if num is None:
        num_np = np.full((boxes_np.shape[0],), boxes_np.shape[1], dtype=np.int32)
    else:
        num_np = np.asarray(num).astype(np.int32).reshape(-1)
    return boxes_np, scores_np, classes_np, num_np


def _map_class_to_category_id(
    raw_class: float, category_ids: list[int], category_id_set: set[int]
) -> int | None:
    value = int(round(float(raw_class)))
    if value in category_id_set:
        return value
    if 0 <= value < len(category_ids):
        return int(category_ids[value])
    if 1 <= value <= len(category_ids):
        return int(category_ids[value - 1])
    return None


def _boxes_to_xywh_pixels(
    boxes_yxyx: np.ndarray,
    *,
    input_size: int,
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    if boxes_yxyx.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = boxes_yxyx.astype(np.float32)
    if float(np.max(boxes)) <= 1.5:
        boxes = boxes * float(input_size)

    y0 = boxes[:, 0]
    x0 = boxes[:, 1]
    y1 = boxes[:, 2]
    x1 = boxes[:, 3]

    scale = min(float(input_size) / float(orig_w), float(input_size) / float(orig_h))
    new_w = float(orig_w) * scale
    new_h = float(orig_h) * scale
    pad_x = (float(input_size) - new_w) / 2.0
    pad_y = (float(input_size) - new_h) / 2.0

    x0 = (x0 - pad_x) / scale
    x1 = (x1 - pad_x) / scale
    y0 = (y0 - pad_y) / scale
    y1 = (y1 - pad_y) / scale

    x0 = np.clip(x0, 0.0, float(orig_w))
    x1 = np.clip(x1, 0.0, float(orig_w))
    y0 = np.clip(y0, 0.0, float(orig_h))
    y1 = np.clip(y1, 0.0, float(orig_h))

    w = np.clip(x1 - x0, 0.0, float(orig_w))
    h = np.clip(y1 - y0, 0.0, float(orig_h))
    return np.stack([x0, y0, w, h], axis=1).astype(np.float32)


def _decode_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _print_debug_io_once(
    *,
    teacher: TeacherModel,
    output_dict: dict[str, Any],
) -> None:
    print(
        "[pseudo-label] teacher signature: "
        f"{teacher.signature_name}, input_dtype={teacher.input_dtype_name}"
    )
    print("[pseudo-label] output tensors:")
    for key, value in sorted(output_dict.items(), key=lambda item: item[0]):
        shape = tuple(np.asarray(value).shape)
        print(f"  - {key}: shape={shape}")


def _write_visual_samples(
    *,
    images_dir: Path,
    image_records: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    out_dir: Path,
    max_images: int,
) -> int:
    by_image: dict[int, list[dict[str, Any]]] = {}
    for det in detections:
        by_image.setdefault(int(det["image_id"]), []).append(det)

    name_by_id = {int(item["id"]): str(item["name"]) for item in categories}
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for image in image_records:
        if written >= max_images:
            break
        image_id = int(image["id"])
        image_dets = by_image.get(image_id)
        if not image_dets:
            continue
        src_path = images_dir / str(image["file_name"])
        if not src_path.is_file():
            continue
        with Image.open(src_path) as pil:
            canvas = pil.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for det in image_dets:
            x, y, w, h = [float(v) for v in det["bbox"]]
            draw.rectangle([(x, y), (x + w, y + h)], outline=(255, 64, 64), width=2)
            cls = name_by_id.get(int(det["category_id"]), str(det["category_id"]))
            text = f"{cls} {float(det['score']):.2f}"
            draw.text((x + 2, max(0.0, y - 12)), text, fill=(255, 64, 64))
        out_path = out_dir / f"{Path(str(image['file_name'])).stem}.jpg"
        canvas.save(out_path, format="JPEG", quality=90)
        written += 1
    return written


def generate_teacher_pseudo_labels(cfg: TeacherPseudoLabelConfig) -> TeacherPseudoLabelArtifacts:
    tf, _ = _ensure_teacher_dependencies()
    teacher_source = (
        str(cfg.teacher_savedmodel) if cfg.teacher_savedmodel is not None else cfg.teacher
    )
    teacher = load_teacher(teacher_source)
    categories = load_coco80_categories()
    allowed_categories = parse_classes_filter(cfg.classes_filter, categories=categories)

    image_paths = _discover_images(cfg.images_dir)
    if not image_paths:
        raise TeacherPseudoLabelConfigError(f"No images found under: {cfg.images_dir}")

    image_paths = sorted(image_paths)
    if cfg.seed != 0:
        rng = np.random.default_rng(seed=cfg.seed)
        perm = rng.permutation(len(image_paths))
        image_paths = [image_paths[int(idx)] for idx in perm]
    if cfg.limit_images is not None:
        image_paths = image_paths[: cfg.limit_images]

    rel_names = [str(path.relative_to(cfg.images_dir).as_posix()) for path in image_paths]
    dataset = _build_tf_dataset(
        tf=tf,
        image_paths=image_paths,
        rel_names=rel_names,
        input_size=cfg.input_size,
        batch_size=cfg.batch_size,
        num_parallel_calls=cfg.num_parallel_calls,
        prefetch_buffer=cfg.prefetch_buffer,
        input_dtype_name=teacher.input_dtype_name,
    )

    category_ids = [int(item["id"]) for item in categories]
    category_id_set = set(category_ids)

    image_records: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    next_image_id = 1
    debug_printed = False

    start = time.perf_counter()
    for batch in dataset:
        outputs = teacher.runner(batch["image"])
        output_dict = _resolve_output(outputs)
        if cfg.debug_io and not debug_printed:
            _print_debug_io_once(teacher=teacher, output_dict=output_dict)
            debug_printed = True

        boxes, scores, classes, num = _extract_detection_arrays(output_dict)
        batch_file_names = [_decode_name(item) for item in np.asarray(batch["file_name"]).tolist()]
        batch_h = np.asarray(batch["orig_h"], dtype=np.int32)
        batch_w = np.asarray(batch["orig_w"], dtype=np.int32)

        for idx, file_name in enumerate(batch_file_names):
            image_id = next_image_id
            next_image_id += 1
            orig_h = int(batch_h[idx])
            orig_w = int(batch_w[idx])
            image_records.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": orig_w,
                    "height": orig_h,
                }
            )

            keep_n = min(int(num[idx]), int(scores.shape[1]))
            if keep_n <= 0:
                continue
            sample_scores = scores[idx, :keep_n].astype(np.float32)
            sample_boxes = boxes[idx, :keep_n, :].astype(np.float32)
            sample_classes = classes[idx, :keep_n].astype(np.float32)

            order = np.argsort(-sample_scores)
            if cfg.max_detections_per_image > 0:
                order = order[: cfg.max_detections_per_image]

            sample_scores = sample_scores[order]
            sample_boxes = sample_boxes[order]
            sample_classes = sample_classes[order]
            sample_xywh = _boxes_to_xywh_pixels(
                sample_boxes,
                input_size=cfg.input_size,
                orig_w=orig_w,
                orig_h=orig_h,
            )

            kept = 0
            for det_idx, score in enumerate(sample_scores.tolist()):
                if score < cfg.score_threshold:
                    continue
                category_id = _map_class_to_category_id(
                    raw_class=float(sample_classes[det_idx]),
                    category_ids=category_ids,
                    category_id_set=category_id_set,
                )
                if category_id is None:
                    continue
                if allowed_categories is not None and category_id not in allowed_categories:
                    continue
                bbox = [float(v) for v in sample_xywh[det_idx].tolist()]
                if bbox[2] <= 0.0 or bbox[3] <= 0.0:
                    continue
                detections.append(
                    {
                        "image_id": image_id,
                        "category_id": int(category_id),
                        "bbox": bbox,
                        "score": float(score),
                    }
                )
                kept += 1
                if kept >= cfg.max_detections_per_image:
                    break
    elapsed = time.perf_counter() - start

    coco = build_pseudo_coco(images=image_records, detections=detections, categories=categories)
    report = build_pseudo_report(
        num_images=len(image_records),
        detections=detections,
        categories=categories,
        total_seconds=elapsed,
        teacher_source=teacher.source,
        batch_size=cfg.batch_size,
        input_size=cfg.input_size,
        score_threshold=cfg.score_threshold,
        max_detections_per_image=cfg.max_detections_per_image,
    )

    if cfg.viz_out_dir is not None:
        written = _write_visual_samples(
            images_dir=cfg.images_dir,
            image_records=image_records,
            detections=detections,
            categories=categories,
            out_dir=cfg.viz_out_dir,
            max_images=cfg.viz_max_images,
        )
        report["visual_samples"] = {"out_dir": str(cfg.viz_out_dir), "images_written": int(written)}

    write_json(cfg.out_path, coco)
    write_json(cfg.report_out_path, report)
    return TeacherPseudoLabelArtifacts(
        pseudo_coco_path=cfg.out_path,
        report_path=cfg.report_out_path,
        images_processed=len(image_records),
        detections_kept=len(detections),
        teacher_source=teacher.source,
        elapsed_seconds=float(elapsed),
    )
