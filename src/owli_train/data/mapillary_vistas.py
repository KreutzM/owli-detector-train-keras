from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from owli_train.data.coco import validate_coco, write_coco
from owli_train.data.split import build_split_coco, write_splits


@dataclass(frozen=True)
class MapillarySplitArtifacts:
    images: int
    annotations: int
    skipped_images_without_labels: int
    skipped_annotations_invalid_bbox: int
    class_counts: dict[str, int]


@dataclass(frozen=True)
class MapillaryImportArtifacts:
    out_dir: Path
    images_dir: Path
    combined_coco_path: Path
    train_coco_path: Path
    val_coco_path: Path
    splits_path: Path
    class_names_path: Path
    qc_report_path: Path
    train: MapillarySplitArtifacts
    val: MapillarySplitArtifacts
    categories: list[str]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain an object: {path}")
    return payload


def _load_label_mapping(map_path: Path) -> tuple[dict[str, str], list[str]]:
    payload = _load_yaml(map_path)
    raw_map = payload.get("map", payload)
    if not isinstance(raw_map, dict):
        raise ValueError("Mapillary label map must resolve to an object mapping source->target labels.")
    mapping = {str(src).strip(): str(dst).strip() for src, dst in raw_map.items()}
    if any(not src or not dst for src, dst in mapping.items()):
        raise ValueError("Mapillary label map contains empty source or target labels.")

    contract_order: list[str] = []
    contract_ref = payload.get("ba_contract")
    if isinstance(contract_ref, str) and contract_ref.strip():
        contract_path = (map_path.parent / contract_ref).resolve()
        if not contract_path.is_file():
            contract_path = Path(contract_ref)
        contract_payload = _load_yaml(contract_path)
        raw_class_names = contract_payload.get("class_names")
        if not isinstance(raw_class_names, list) or not raw_class_names:
            raise ValueError("Referenced ba_contract must contain a non-empty class_names list.")
        contract_order = [str(item).strip() for item in raw_class_names]

    return mapping, contract_order


def _resolve_category_order(mapped_targets: set[str], contract_order: list[str]) -> list[str]:
    if contract_order:
        ordered = [name for name in contract_order if name in mapped_targets]
        if ordered:
            return ordered
    return sorted(mapped_targets)


def _load_panoptic_payload(split_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[int, str]]:
    json_path = split_root / "panoptic" / "panoptic_2018.json"
    if not json_path.is_file():
        raise ValueError(f"Missing Mapillary panoptic JSON: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Panoptic JSON must be an object: {json_path}")

    raw_images = payload.get("images")
    raw_annotations = payload.get("annotations")
    raw_categories = payload.get("categories")
    if not isinstance(raw_images, list) or not isinstance(raw_annotations, list) or not isinstance(raw_categories, list):
        raise ValueError(f"Panoptic JSON missing required lists: {json_path}")

    images_by_id: dict[str, dict[str, Any]] = {}
    for item in raw_images:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid panoptic image entry in {json_path}")
        image_id = str(item.get("id", "")).strip()
        file_name = str(item.get("file_name", "")).strip()
        if not image_id or not file_name:
            raise ValueError(f"Panoptic image entry missing id/file_name in {json_path}")
        images_by_id[image_id] = item

    annotations_by_image_id: dict[str, dict[str, Any]] = {}
    for item in raw_annotations:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid panoptic annotation entry in {json_path}")
        image_id = str(item.get("image_id", "")).strip()
        if not image_id:
            raise ValueError(f"Panoptic annotation missing image_id in {json_path}")
        annotations_by_image_id[image_id] = item

    category_name_by_id: dict[int, str] = {}
    for item in raw_categories:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid panoptic category entry in {json_path}")
        try:
            category_id = int(item["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Panoptic category missing integer id in {json_path}") from exc
        source_name = str(item.get("supercategory", "")).strip()
        if not source_name:
            raise ValueError(f"Panoptic category missing supercategory in {json_path}")
        category_name_by_id[category_id] = source_name

    return images_by_id, annotations_by_image_id, category_name_by_id


def _scaled_dimensions(width: int, height: int, max_long_side: int) -> tuple[int, int, float]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if max_long_side <= 0:
        raise ValueError("max_long_side must be > 0.")
    long_side = max(width, height)
    if long_side <= max_long_side:
        return width, height, 1.0
    scale = float(max_long_side) / float(long_side)
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    return scaled_width, scaled_height, scale


def _scale_bbox(bbox: list[float], scale: float, max_width: int, max_height: int) -> list[float] | None:
    if len(bbox) != 4:
        return None
    x, y, width, height = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if width <= 0.0 or height <= 0.0:
        return None

    scaled_x = max(0.0, min(float(max_width), x * scale))
    scaled_y = max(0.0, min(float(max_height), y * scale))
    scaled_w = width * scale
    scaled_h = height * scale

    if scaled_x + scaled_w > float(max_width):
        scaled_w = float(max_width) - scaled_x
    if scaled_y + scaled_h > float(max_height):
        scaled_h = float(max_height) - scaled_y
    if scaled_w <= 0.0 or scaled_h <= 0.0:
        return None
    return [round(scaled_x, 4), round(scaled_y, 4), round(scaled_w, 4), round(scaled_h, 4)]


def _write_resized_image(
    source_path: Path,
    dest_path: Path,
    *,
    resized_width: int,
    resized_height: int,
) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        if image.width == resized_width and image.height == resized_height:
            shutil.copy2(source_path, dest_path)
            return

        resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        save_kwargs: dict[str, Any] = {}
        if dest_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = 90
        resized.save(dest_path, **save_kwargs)


def _convert_split(
    *,
    split_name: str,
    split_root: Path,
    out_images_root: Path,
    label_mapping: dict[str, str],
    category_id_by_name: dict[str, int],
    max_long_side: int,
    starting_image_id: int,
    starting_annotation_id: int,
    limit_images: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], MapillarySplitArtifacts, int, int]:
    images_by_id, annotations_by_image_id, source_name_by_category_id = _load_panoptic_payload(split_root)
    raw_image_ids = sorted(images_by_id)
    if limit_images is not None:
        raw_image_ids = raw_image_ids[:limit_images]

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    kept_image_ids: list[int] = []
    class_counts = {name: 0 for name in category_id_by_name}
    skipped_images_without_labels = 0
    skipped_annotations_invalid_bbox = 0
    next_image_id = starting_image_id
    next_annotation_id = starting_annotation_id

    for source_image_id in raw_image_ids:
        image_info = images_by_id[source_image_id]
        image_annotations = annotations_by_image_id.get(source_image_id)
        if image_annotations is None:
            skipped_images_without_labels += 1
            continue

        source_file_name = str(image_info["file_name"])
        source_image_path = split_root / "images" / source_file_name
        if not source_image_path.is_file():
            raise ValueError(f"Referenced Mapillary image is missing: {source_image_path}")

        width = int(image_info["width"])
        height = int(image_info["height"])
        resized_width, resized_height, scale = _scaled_dimensions(width, height, max_long_side)

        kept_annotations_for_image: list[dict[str, Any]] = []
        for segment in image_annotations.get("segments_info", []):
            if not isinstance(segment, dict):
                continue
            try:
                category_id = int(segment["category_id"])
            except (KeyError, TypeError, ValueError):
                continue
            source_class = source_name_by_category_id.get(category_id)
            if source_class is None:
                continue
            target_class = label_mapping.get(source_class)
            if target_class is None:
                continue

            bbox = segment.get("bbox")
            if not isinstance(bbox, list):
                skipped_annotations_invalid_bbox += 1
                continue
            scaled_bbox = _scale_bbox(
                bbox=bbox,
                scale=scale,
                max_width=resized_width,
                max_height=resized_height,
            )
            if scaled_bbox is None:
                skipped_annotations_invalid_bbox += 1
                continue

            category_id_out = category_id_by_name[target_class]
            area = round(scaled_bbox[2] * scaled_bbox[3], 4)
            if area <= 0.0:
                skipped_annotations_invalid_bbox += 1
                continue

            kept_annotations_for_image.append(
                {
                    "id": next_annotation_id,
                    "image_id": next_image_id,
                    "category_id": category_id_out,
                    "bbox": scaled_bbox,
                    "area": area,
                    "iscrowd": int(segment.get("iscrowd", 0)),
                }
            )
            next_annotation_id += 1
            class_counts[target_class] += 1

        if not kept_annotations_for_image:
            skipped_images_without_labels += 1
            continue

        rel_file_name = Path(split_name) / source_file_name
        dest_image_path = out_images_root / rel_file_name
        _write_resized_image(
            source_path=source_image_path,
            dest_path=dest_image_path,
            resized_width=resized_width,
            resized_height=resized_height,
        )

        images.append(
            {
                "id": next_image_id,
                "file_name": rel_file_name.as_posix(),
                "width": resized_width,
                "height": resized_height,
                "source_image_id": source_image_id,
                "source_split": split_name,
            }
        )
        annotations.extend(kept_annotations_for_image)
        kept_image_ids.append(next_image_id)
        next_image_id += 1

    return (
        images,
        annotations,
        kept_image_ids,
        MapillarySplitArtifacts(
            images=len(images),
            annotations=len(annotations),
            skipped_images_without_labels=skipped_images_without_labels,
            skipped_annotations_invalid_bbox=skipped_annotations_invalid_bbox,
            class_counts={k: v for k, v in class_counts.items() if v > 0},
        ),
        next_image_id,
        next_annotation_id,
    )


def import_mapillary_vistas_to_coco(
    *,
    mapillary_dir: Path,
    out_dir: Path,
    label_map_path: Path,
    max_long_side: int = 1600,
    limit_images_per_split: int | None = None,
) -> MapillaryImportArtifacts:
    source_root = Path(mapillary_dir)
    if not source_root.is_dir():
        raise ValueError(f"--mapillary-dir was not found: {source_root}")

    label_mapping, contract_order = _load_label_mapping(label_map_path)
    if not label_mapping:
        raise ValueError("Resolved zero source->target mappings from --label-map.")

    mapped_targets = set(label_mapping.values())
    category_names = _resolve_category_order(mapped_targets=mapped_targets, contract_order=contract_order)
    category_id_by_name = {name: idx + 1 for idx, name in enumerate(category_names)}
    categories = [{"id": idx, "name": name} for name, idx in category_id_by_name.items()]

    out_root = Path(out_dir)
    images_root = out_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    train_images, train_annotations, train_ids, train_stats, next_image_id, next_annotation_id = _convert_split(
        split_name="training",
        split_root=source_root / "training",
        out_images_root=images_root,
        label_mapping=label_mapping,
        category_id_by_name=category_id_by_name,
        max_long_side=max_long_side,
        starting_image_id=1,
        starting_annotation_id=1,
        limit_images=limit_images_per_split,
    )
    val_images, val_annotations, val_ids, val_stats, _, _ = _convert_split(
        split_name="validation",
        split_root=source_root / "validation",
        out_images_root=images_root,
        label_mapping=label_mapping,
        category_id_by_name=category_id_by_name,
        max_long_side=max_long_side,
        starting_image_id=next_image_id,
        starting_annotation_id=next_annotation_id,
        limit_images=limit_images_per_split,
    )

    combined = {
        "images": train_images + val_images,
        "annotations": train_annotations + val_annotations,
        "categories": categories,
    }
    validate_coco(combined, images_dir=images_root)

    combined_path = write_coco(out_root / "instances_ba_v1.coco.json", combined)
    train_path = write_coco(out_root / "annotations_train.coco.json", build_split_coco(combined, train_ids))
    val_path = write_coco(out_root / "annotations_val.coco.json", build_split_coco(combined, val_ids))

    splits_path = write_splits(out_root, {"train": train_ids, "val": val_ids, "test": []})
    class_names_path = out_root / "class_names.json"
    class_names_path.write_text(
        json.dumps(
            {
                "class_names": category_names,
                "source_label_map": str(label_map_path),
                "source_dataset": str(source_root),
                "max_long_side": int(max_long_side),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    qc_report_path = out_root / "qc_report.json"
    qc_report_path.write_text(
        json.dumps(
            {
                "source_dataset": str(source_root),
                "label_map": str(label_map_path),
                "max_long_side": int(max_long_side),
                "limit_images_per_split": int(limit_images_per_split)
                if limit_images_per_split is not None
                else None,
                "categories": category_names,
                "splits": {
                    "training": {
                        "images": train_stats.images,
                        "annotations": train_stats.annotations,
                        "skipped_images_without_labels": train_stats.skipped_images_without_labels,
                        "skipped_annotations_invalid_bbox": train_stats.skipped_annotations_invalid_bbox,
                        "class_counts": train_stats.class_counts,
                    },
                    "validation": {
                        "images": val_stats.images,
                        "annotations": val_stats.annotations,
                        "skipped_images_without_labels": val_stats.skipped_images_without_labels,
                        "skipped_annotations_invalid_bbox": val_stats.skipped_annotations_invalid_bbox,
                        "class_counts": val_stats.class_counts,
                    },
                },
                "combined": {
                    "images": len(combined["images"]),
                    "annotations": len(combined["annotations"]),
                    "categories": len(combined["categories"]),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return MapillaryImportArtifacts(
        out_dir=out_root,
        images_dir=images_root,
        combined_coco_path=combined_path,
        train_coco_path=train_path,
        val_coco_path=val_path,
        splits_path=splits_path,
        class_names_path=class_names_path,
        qc_report_path=qc_report_path,
        train=train_stats,
        val=val_stats,
        categories=category_names,
    )
