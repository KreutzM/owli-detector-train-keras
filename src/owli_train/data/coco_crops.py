from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from owli_train.data.coco import load_coco, validate_coco, write_coco


@dataclass(frozen=True)
class CocoCropConfig:
    source_coco: Path
    source_images_dir: Path
    out_dir: Path
    target_classes: list[str]
    allowed_source_prefixes: list[str]
    max_bbox_min_side: float
    max_bbox_area_ratio: float
    max_bbox_short_side_ratio: float
    max_crops_per_class: int
    max_crops_per_image: int
    context_scale: float
    min_crop_size: int
    max_crop_size: int
    min_retained_area_ratio: float
    min_retained_bbox_min_side: float
    file_name_prefix: str


@dataclass(frozen=True)
class CocoCropArtifacts:
    out_dir: Path
    images_dir: Path
    coco_path: Path
    class_names_path: Path
    qc_report_path: Path
    images: int
    annotations: int
    categories: int


@dataclass(frozen=True)
class CropCandidate:
    image_id: int
    annotation_id: int
    category_id: int
    category_name: str
    source_prefix: str
    source_file_name: str
    source_bbox: tuple[float, float, float, float]
    image_width: int
    image_height: int
    source_min_side: float
    source_area_ratio: float
    source_short_side_ratio: float


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain an object: {path}")
    return payload


def _resolve_path(value: Any, *, config_path: Path, expect_dir: bool = False) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Required config path value must be set.")
    path = Path(raw)
    resolved = path if path.is_absolute() else (config_path.parent / path).resolve()
    if expect_dir:
        if not resolved.is_dir():
            raise ValueError(f"Directory does not exist: {resolved}")
    else:
        if not resolved.is_file():
            raise ValueError(f"File does not exist: {resolved}")
    return resolved


def _resolve_out_dir(value: Any, *, config_path: Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Config must set out_dir.")
    path = Path(raw)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _coerce_non_negative_float(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number >= 0.") from exc
    if number < 0.0:
        raise ValueError(f"{label} must be >= 0.")
    return number


def _coerce_ratio(value: Any, *, label: str) -> float:
    number = _coerce_non_negative_float(value, label=label)
    if number > 1.0:
        raise ValueError(f"{label} must be in [0.0, 1.0].")
    return number


def _coerce_positive_int(value: Any, *, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer > 0.") from exc
    if number <= 0:
        raise ValueError(f"{label} must be > 0.")
    return number


def _load_string_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list.")
    items = [str(item).strip() for item in value]
    if any(not item for item in items):
        raise ValueError(f"{label} must not contain empty values.")
    return items


def load_coco_crop_config(path: Path) -> CocoCropConfig:
    payload = _load_yaml(path)
    selection = payload.get("selection", {})
    if not isinstance(selection, dict):
        raise ValueError("selection must be an object.")
    crop = payload.get("crop", {})
    if not isinstance(crop, dict):
        raise ValueError("crop must be an object.")

    file_name_prefix = str(crop.get("file_name_prefix", "stage3_crops")).strip().strip("/")
    if not file_name_prefix:
        raise ValueError("crop.file_name_prefix must be non-empty.")

    min_crop_size = _coerce_positive_int(crop.get("min_size", 256), label="crop.min_size")
    max_crop_size = _coerce_positive_int(crop.get("max_size", 512), label="crop.max_size")
    if max_crop_size < min_crop_size:
        raise ValueError("crop.max_size must be >= crop.min_size.")

    return CocoCropConfig(
        source_coco=_resolve_path(payload.get("source_coco"), config_path=path),
        source_images_dir=_resolve_path(
            payload.get("source_images_dir"),
            config_path=path,
            expect_dir=True,
        ),
        out_dir=_resolve_out_dir(payload.get("out_dir"), config_path=path),
        target_classes=_load_string_list(selection.get("target_classes"), label="selection.target_classes"),
        allowed_source_prefixes=_load_string_list(
            selection.get("allowed_source_prefixes"),
            label="selection.allowed_source_prefixes",
        ),
        max_bbox_min_side=_coerce_non_negative_float(
            selection.get("max_bbox_min_side", 48),
            label="selection.max_bbox_min_side",
        ),
        max_bbox_area_ratio=_coerce_ratio(
            selection.get("max_bbox_area_ratio", 0.01),
            label="selection.max_bbox_area_ratio",
        ),
        max_bbox_short_side_ratio=_coerce_ratio(
            selection.get("max_bbox_short_side_ratio", 0.05),
            label="selection.max_bbox_short_side_ratio",
        ),
        max_crops_per_class=_coerce_positive_int(
            selection.get("max_crops_per_class", 200),
            label="selection.max_crops_per_class",
        ),
        max_crops_per_image=_coerce_positive_int(
            selection.get("max_crops_per_image", 1),
            label="selection.max_crops_per_image",
        ),
        context_scale=_coerce_non_negative_float(
            crop.get("context_scale", 4.0),
            label="crop.context_scale",
        ),
        min_crop_size=min_crop_size,
        max_crop_size=max_crop_size,
        min_retained_area_ratio=_coerce_ratio(
            crop.get("min_retained_area_ratio", 0.5),
            label="crop.min_retained_area_ratio",
        ),
        min_retained_bbox_min_side=_coerce_non_negative_float(
            crop.get("min_retained_bbox_min_side", 4),
            label="crop.min_retained_bbox_min_side",
        ),
        file_name_prefix=file_name_prefix,
    )


def _write_class_names(path: Path, categories: list[dict[str, Any]]) -> Path:
    ordered = sorted(categories, key=lambda cat: int(cat["id"]))
    payload = {
        "class_names": [str(cat["name"]) for cat in ordered],
        "category_ids": [int(cat["id"]) for cat in ordered],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _extract_source_prefix(file_name: str) -> str:
    normalized = str(file_name).replace("\\", "/").strip("/")
    if not normalized:
        return ""
    return normalized.split("/", 1)[0]


def _is_crop_worthy(
    *,
    bbox_min_side: float,
    bbox_area_ratio: float,
    bbox_short_side_ratio: float,
    config: CocoCropConfig,
) -> bool:
    return (
        bbox_min_side <= config.max_bbox_min_side
        or bbox_area_ratio <= config.max_bbox_area_ratio
        or bbox_short_side_ratio <= config.max_bbox_short_side_ratio
    )


def _build_crop_candidates(
    *,
    coco: dict[str, Any],
    config: CocoCropConfig,
) -> tuple[dict[str, list[CropCandidate]], dict[str, int]]:
    images_by_id = {int(image["id"]): dict(image) for image in coco["images"]}
    category_name_by_id = {int(category["id"]): str(category["name"]) for category in coco["categories"]}
    source_priority = {name: idx for idx, name in enumerate(config.allowed_source_prefixes)}
    unknown_target_classes = sorted(set(config.target_classes) - set(category_name_by_id.values()))
    if unknown_target_classes:
        preview = ", ".join(unknown_target_classes)
        raise ValueError(f"selection.target_classes are not present in source COCO: {preview}")

    candidates_by_class: dict[str, list[CropCandidate]] = {name: [] for name in config.target_classes}
    skipped = Counter()

    for annotation in coco["annotations"]:
        image_id = int(annotation["image_id"])
        image = images_by_id[image_id]
        category_name = category_name_by_id[int(annotation["category_id"])]
        if category_name not in candidates_by_class:
            skipped["non_target_class"] += 1
            continue

        source_prefix = _extract_source_prefix(str(image.get("file_name", "")))
        if source_prefix not in source_priority:
            skipped["disallowed_source_prefix"] += 1
            continue

        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            skipped["invalid_bbox"] += 1
            continue
        x, y, width, height = [float(value) for value in bbox]
        image_width = int(image.get("width") or 0)
        image_height = int(image.get("height") or 0)
        if image_width <= 0 or image_height <= 0:
            skipped["missing_image_size"] += 1
            continue

        bbox_min_side = min(width, height)
        bbox_area_ratio = (width * height) / float(image_width * image_height)
        bbox_short_side_ratio = bbox_min_side / float(min(image_width, image_height))
        if not _is_crop_worthy(
            bbox_min_side=bbox_min_side,
            bbox_area_ratio=bbox_area_ratio,
            bbox_short_side_ratio=bbox_short_side_ratio,
            config=config,
        ):
            skipped["not_crop_worthy"] += 1
            continue

        candidates_by_class[category_name].append(
            CropCandidate(
                image_id=image_id,
                annotation_id=int(annotation["id"]),
                category_id=int(annotation["category_id"]),
                category_name=category_name,
                source_prefix=source_prefix,
                source_file_name=str(image["file_name"]),
                source_bbox=(x, y, width, height),
                image_width=image_width,
                image_height=image_height,
                source_min_side=bbox_min_side,
                source_area_ratio=bbox_area_ratio,
                source_short_side_ratio=bbox_short_side_ratio,
            )
        )

    for candidates in candidates_by_class.values():
        candidates.sort(
            key=lambda item: (
                source_priority[item.source_prefix],
                item.source_short_side_ratio,
                item.source_area_ratio,
                item.source_min_side,
                item.image_id,
                item.annotation_id,
            )
        )

    return candidates_by_class, dict(skipped)


def _class_selection_order(
    *,
    target_classes: list[str],
    candidates_by_class: dict[str, list[CropCandidate]],
) -> list[str]:
    image_counts = {
        class_name: len({candidate.image_id for candidate in candidates})
        for class_name, candidates in candidates_by_class.items()
    }
    return sorted(
        target_classes,
        key=lambda class_name: (image_counts[class_name], target_classes.index(class_name)),
    )


def _compute_crop_box(candidate: CropCandidate, config: CocoCropConfig) -> tuple[int, int, int, int]:
    _, _, width, height = candidate.source_bbox
    desired_side = max(
        config.min_crop_size,
        int(math.ceil(max(width, height) * config.context_scale)),
    )
    max_side = min(candidate.image_width, candidate.image_height, config.max_crop_size)
    side = min(desired_side, max_side)
    if side <= 0:
        raise ValueError("Computed crop side must be > 0.")

    x, y, bbox_width, bbox_height = candidate.source_bbox
    center_x = x + (bbox_width / 2.0)
    center_y = y + (bbox_height / 2.0)

    left = int(round(center_x - (side / 2.0)))
    top = int(round(center_y - (side / 2.0)))
    max_left = candidate.image_width - side
    max_top = candidate.image_height - side
    left = min(max(left, 0), max_left)
    top = min(max(top, 0), max_top)
    return left, top, side, side


def _clip_bbox_to_crop(
    *,
    bbox: tuple[float, float, float, float],
    crop_box: tuple[int, int, int, int],
) -> tuple[float, float, float, float] | None:
    crop_left, crop_top, crop_width, crop_height = crop_box
    crop_right = crop_left + crop_width
    crop_bottom = crop_top + crop_height

    bbox_left, bbox_top, bbox_width, bbox_height = bbox
    bbox_right = bbox_left + bbox_width
    bbox_bottom = bbox_top + bbox_height

    clipped_left = max(bbox_left, crop_left)
    clipped_top = max(bbox_top, crop_top)
    clipped_right = min(bbox_right, crop_right)
    clipped_bottom = min(bbox_bottom, crop_bottom)
    clipped_width = clipped_right - clipped_left
    clipped_height = clipped_bottom - clipped_top
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        return None

    return (
        clipped_left - crop_left,
        clipped_top - crop_top,
        clipped_width,
        clipped_height,
    )


def _bbox_iou_xywh(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    left_x, left_y, left_w, left_h = left
    right_x, right_y, right_w, right_h = right
    left_x1 = left_x + left_w
    left_y1 = left_y + left_h
    right_x1 = right_x + right_w
    right_y1 = right_y + right_h

    inter_x0 = max(left_x, right_x)
    inter_y0 = max(left_y, right_y)
    inter_x1 = min(left_x1, right_x1)
    inter_y1 = min(left_y1, right_y1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    left_area = left_w * left_h
    right_area = right_w * right_h
    denom = left_area + right_area - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def _dedupe_same_class_annotations(
    annotations: list[dict[str, Any]],
    *,
    same_class_iou: float = 0.75,
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    dropped = 0
    for annotation in annotations:
        bbox = tuple(float(value) for value in annotation["bbox"])
        category_id = int(annotation["category_id"])
        duplicate = False
        for prior in kept:
            if int(prior["category_id"]) != category_id:
                continue
            prior_bbox = tuple(float(value) for value in prior["bbox"])
            if _bbox_iou_xywh(bbox, prior_bbox) >= same_class_iou:
                duplicate = True
                dropped += 1
                break
        if not duplicate:
            kept.append(annotation)
    return kept, dropped


def _build_crop_file_name(
    *,
    candidate: CropCandidate,
    file_name_prefix: str,
    image_suffix: str,
) -> str:
    safe_suffix = image_suffix if image_suffix else ".jpg"
    return (
        f"{file_name_prefix}/{candidate.category_name}/{candidate.source_prefix}/"
        f"{candidate.image_id:07d}_ann{candidate.annotation_id:07d}{safe_suffix.lower()}"
    )


def export_coco_crops_with_config(config_path: Path) -> CocoCropArtifacts:
    config = load_coco_crop_config(config_path)
    coco = load_coco(config.source_coco)
    _ = validate_coco(coco, images_dir=config.source_images_dir)

    candidates_by_class, skipped_candidates = _build_crop_candidates(coco=coco, config=config)
    class_selection_order = _class_selection_order(
        target_classes=config.target_classes,
        candidates_by_class=candidates_by_class,
    )

    category_name_by_id = {int(category["id"]): str(category["name"]) for category in coco["categories"]}
    annotations_by_image_id: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image_id.setdefault(int(annotation["image_id"]), []).append(dict(annotation))

    selected_candidates: list[CropCandidate] = []
    selected_images_per_class = Counter()
    selected_candidates_per_source = Counter()
    selected_crops_per_image = Counter()
    selected_candidate_keys: set[tuple[int, int]] = set()

    for class_name in class_selection_order:
        for candidate in candidates_by_class[class_name]:
            if selected_images_per_class[class_name] >= config.max_crops_per_class:
                break
            if selected_crops_per_image[candidate.image_id] >= config.max_crops_per_image:
                continue
            candidate_key = (candidate.image_id, candidate.annotation_id)
            if candidate_key in selected_candidate_keys:
                continue
            selected_candidate_keys.add(candidate_key)
            selected_candidates.append(candidate)
            selected_images_per_class[class_name] += 1
            selected_candidates_per_source[candidate.source_prefix] += 1
            selected_crops_per_image[candidate.image_id] += 1

    if not selected_candidates:
        raise ValueError("No crop candidates were selected. Loosen the crop selection thresholds.")

    out_images_dir = config.out_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    output_images: list[dict[str, Any]] = []
    output_annotations: list[dict[str, Any]] = []
    selected_annotation_counts = Counter()
    selected_crop_targets_by_source = Counter()
    written_crops_by_class = Counter()
    written_crops_by_source = Counter()
    dropped_duplicate_same_class_annotations = 0
    sample_crops: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1

    for candidate in selected_candidates:
        crop_box = _compute_crop_box(candidate, config)
        source_path = config.source_images_dir / candidate.source_file_name
        image_suffix = Path(candidate.source_file_name).suffix or ".jpg"
        crop_file_name = _build_crop_file_name(
            candidate=candidate,
            file_name_prefix=config.file_name_prefix,
            image_suffix=image_suffix,
        )
        crop_path = out_images_dir / crop_file_name
        crop_path.parent.mkdir(parents=True, exist_ok=True)

        retained_annotations: list[dict[str, Any]] = []
        for source_annotation in annotations_by_image_id.get(candidate.image_id, []):
            source_bbox_raw = source_annotation.get("bbox")
            if not isinstance(source_bbox_raw, list) or len(source_bbox_raw) != 4:
                continue
            source_bbox = tuple(float(value) for value in source_bbox_raw)
            clipped_bbox = _clip_bbox_to_crop(bbox=source_bbox, crop_box=crop_box)
            if clipped_bbox is None:
                continue

            _, _, clipped_width, clipped_height = clipped_bbox
            source_area = max(0.0, source_bbox[2] * source_bbox[3])
            clipped_area = clipped_width * clipped_height
            if source_area <= 0.0:
                continue
            if clipped_area / source_area < config.min_retained_area_ratio:
                continue
            if min(clipped_width, clipped_height) < config.min_retained_bbox_min_side:
                continue

            retained_annotation = dict(source_annotation)
            retained_annotation["image_id"] = next_image_id
            retained_annotation["bbox"] = [
                round(clipped_bbox[0], 3),
                round(clipped_bbox[1], 3),
                round(clipped_width, 3),
                round(clipped_height, 3),
            ]
            retained_annotation["area"] = round(clipped_area, 3)
            retained_annotations.append(retained_annotation)

        target_annotation_retained = any(
            int(annotation.get("category_id")) == candidate.category_id for annotation in retained_annotations
        )
        if not retained_annotations or not target_annotation_retained:
            continue

        retained_annotations, dropped_duplicates = _dedupe_same_class_annotations(retained_annotations)
        dropped_duplicate_same_class_annotations += dropped_duplicates
        target_annotation_retained = any(
            int(annotation.get("category_id")) == candidate.category_id for annotation in retained_annotations
        )
        if not retained_annotations or not target_annotation_retained:
            continue

        for retained_annotation in retained_annotations:
            retained_annotation["id"] = next_annotation_id
            next_annotation_id += 1
        output_annotations.extend(retained_annotations)
        for retained_annotation in retained_annotations:
            selected_annotation_counts[
                category_name_by_id[int(retained_annotation["category_id"])]
            ] += 1

        crop_left, crop_top, crop_width, crop_height = crop_box
        with Image.open(source_path) as source_image:
            cropped = source_image.crop(
                (
                    crop_left,
                    crop_top,
                    crop_left + crop_width,
                    crop_top + crop_height,
                )
            )
            save_image = cropped
            if image_suffix.lower() in {".jpg", ".jpeg"} and save_image.mode not in {"RGB", "L"}:
                save_image = cropped.convert("RGB")
            save_image.save(crop_path)

        output_images.append(
            {
                "id": next_image_id,
                "file_name": crop_file_name,
                "width": crop_width,
                "height": crop_height,
                "source_image_id": candidate.image_id,
                "source_file_name": candidate.source_file_name,
                "source_prefix": candidate.source_prefix,
                "crop_target_annotation_id": candidate.annotation_id,
                "crop_target_class": candidate.category_name,
                "crop_box": [crop_left, crop_top, crop_width, crop_height],
            }
        )
        written_crops_by_class[candidate.category_name] += 1
        written_crops_by_source[candidate.source_prefix] += 1
        selected_crop_targets_by_source[f"{candidate.source_prefix}:{candidate.category_name}"] += 1
        if len(sample_crops) < 10:
            sample_crops.append(
                {
                    "file_name": crop_file_name,
                    "source_file_name": candidate.source_file_name,
                    "source_image_id": candidate.image_id,
                    "source_annotation_id": candidate.annotation_id,
                    "target_class": candidate.category_name,
                    "source_bbox": [round(value, 3) for value in candidate.source_bbox],
                    "crop_box": [crop_left, crop_top, crop_width, crop_height],
                }
            )
        next_image_id += 1

    if not output_images:
        raise ValueError("No crop images were written after retained-annotation filtering.")

    present_category_ids = {int(annotation["category_id"]) for annotation in output_annotations}
    output_categories = [
        dict(category)
        for category in sorted(coco["categories"], key=lambda item: int(item["id"]))
        if int(category["id"]) in present_category_ids
    ]

    out_coco: dict[str, Any] = {
        "images": output_images,
        "annotations": output_annotations,
        "categories": output_categories,
    }
    for extra_key in ("info", "licenses"):
        if extra_key in coco:
            out_coco[extra_key] = coco[extra_key]

    config.out_dir.mkdir(parents=True, exist_ok=True)
    coco_path = write_coco(config.out_dir / "instances_ba_v1.coco.json", out_coco)
    class_names_path = _write_class_names(config.out_dir / "class_names.json", output_categories)

    qc_report = {
        "config_path": str(config_path),
        "source_coco": str(config.source_coco),
        "source_images_dir": str(config.source_images_dir),
        "selection": {
            "target_classes": config.target_classes,
            "allowed_source_prefixes": config.allowed_source_prefixes,
            "max_bbox_min_side": config.max_bbox_min_side,
            "max_bbox_area_ratio": config.max_bbox_area_ratio,
            "max_bbox_short_side_ratio": config.max_bbox_short_side_ratio,
            "max_crops_per_class": config.max_crops_per_class,
            "max_crops_per_image": config.max_crops_per_image,
        },
        "crop": {
            "context_scale": config.context_scale,
            "min_size": config.min_crop_size,
            "max_size": config.max_crop_size,
            "min_retained_area_ratio": config.min_retained_area_ratio,
            "min_retained_bbox_min_side": config.min_retained_bbox_min_side,
            "file_name_prefix": config.file_name_prefix,
        },
        "input": {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
            "categories": len(coco["categories"]),
        },
        "output": {
            "images": len(output_images),
            "annotations": len(output_annotations),
            "categories": len(output_categories),
        },
        "class_selection_order": class_selection_order,
        "skipped_candidates": skipped_candidates,
        "dropped_duplicate_same_class_annotations": dropped_duplicate_same_class_annotations,
        "candidate_annotations_by_class": {
            class_name: len(candidates)
            for class_name, candidates in candidates_by_class.items()
            if candidates
        },
        "selected_crops_by_class": {
            class_name: written_crops_by_class[class_name]
            for class_name in config.target_classes
            if written_crops_by_class[class_name] > 0
        },
        "selected_crops_by_source": {
            source_name: written_crops_by_source[source_name]
            for source_name in config.allowed_source_prefixes
            if written_crops_by_source[source_name] > 0
        },
        "selected_candidate_crops_by_source": {
            source_name: selected_candidates_per_source[source_name]
            for source_name in config.allowed_source_prefixes
            if selected_candidates_per_source[source_name] > 0
        },
        "selected_crop_targets_by_source": dict(sorted(selected_crop_targets_by_source.items())),
        "selected_annotation_counts": {
            class_name: selected_annotation_counts[class_name]
            for class_name in (
                str(category["name"]) for category in sorted(output_categories, key=lambda item: int(item["id"]))
            )
            if selected_annotation_counts[class_name] > 0
        },
        "sample_crops": sample_crops,
    }
    qc_report_path = config.out_dir / "qc_report.json"
    qc_report_path.write_text(json.dumps(qc_report, indent=2), encoding="utf-8")

    _ = validate_coco(out_coco, images_dir=out_images_dir)

    return CocoCropArtifacts(
        out_dir=config.out_dir,
        images_dir=out_images_dir,
        coco_path=coco_path,
        class_names_path=class_names_path,
        qc_report_path=qc_report_path,
        images=len(output_images),
        annotations=len(output_annotations),
        categories=len(output_categories),
    )
