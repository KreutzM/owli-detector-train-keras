from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CocoSummary:
    images: int
    annotations: int
    categories: int
    category_names: list[str]


def load_coco(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("COCO JSON must be an object at top-level.")
    return obj


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer-like value.") from exc


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object.")
    return value


def _collect_unique_ids(items: list[dict[str, Any]], item_name: str) -> set[int]:
    seen: set[int] = set()
    for idx, item in enumerate(items):
        item_map = _require_mapping(item, f"{item_name}[{idx}]")
        if "id" not in item_map:
            raise ValueError(f"{item_name}[{idx}] is missing required key 'id'.")
        item_id = _coerce_int(item_map["id"], f"{item_name}[{idx}].id")
        if item_id in seen:
            raise ValueError(f"{item_name} contains duplicate id: {item_id}")
        seen.add(item_id)
    return seen


def validate_coco(obj: dict[str, Any], images_dir: str | Path | None = None) -> CocoSummary:
    for key in ("images", "annotations", "categories"):
        if key not in obj:
            raise ValueError(f"Missing COCO key: {key}")
        if not isinstance(obj[key], list):
            raise ValueError(f"COCO key must be a list: {key}")
    if len(obj["categories"]) == 0:
        raise ValueError("COCO categories is empty.")

    image_ids = _collect_unique_ids(obj["images"], "images")
    _ = _collect_unique_ids(obj["annotations"], "annotations")
    category_ids = _collect_unique_ids(obj["categories"], "categories")

    cat_names: list[str] = []
    for idx, category in enumerate(obj["categories"]):
        c = _require_mapping(category, f"categories[{idx}]")
        if "name" not in c:
            raise ValueError(f"categories[{idx}] is missing required key 'name'.")
        cat_name = str(c["name"]).strip()
        if not cat_name:
            raise ValueError(f"categories[{idx}].name must be non-empty.")
        cat_names.append(cat_name)

    images_root = Path(images_dir) if images_dir is not None else None
    for idx, image in enumerate(obj["images"]):
        img = _require_mapping(image, f"images[{idx}]")
        if images_root is not None:
            file_name = img.get("file_name")
            if not isinstance(file_name, str) or not file_name.strip():
                raise ValueError(f"images[{idx}] must include non-empty 'file_name'.")
            image_path = images_root / file_name
            if not image_path.is_file():
                raise ValueError(f"Referenced image file does not exist: {image_path}")

    for idx, annotation in enumerate(obj["annotations"]):
        ann = _require_mapping(annotation, f"annotations[{idx}]")
        for required in ("image_id", "category_id", "bbox"):
            if required not in ann:
                raise ValueError(f"annotations[{idx}] is missing required key '{required}'.")

        ann_image_id = _coerce_int(ann["image_id"], f"annotations[{idx}].image_id")
        if ann_image_id not in image_ids:
            raise ValueError(f"annotations[{idx}] references unknown image_id: {ann_image_id}")

        ann_category_id = _coerce_int(ann["category_id"], f"annotations[{idx}].category_id")
        if ann_category_id not in category_ids:
            raise ValueError(
                f"annotations[{idx}] references unknown category_id: {ann_category_id}"
            )

        bbox = ann["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"annotations[{idx}].bbox must be a list with 4 numbers.")
        if not all(_is_number(v) for v in bbox):
            raise ValueError(f"annotations[{idx}].bbox must contain only numeric values.")
        width = float(bbox[2])
        height = float(bbox[3])
        if width <= 0 or height <= 0:
            raise ValueError(f"annotations[{idx}].bbox width and height must be > 0.")

    return CocoSummary(
        images=len(obj["images"]),
        annotations=len(obj["annotations"]),
        categories=len(obj["categories"]),
        category_names=sorted(set(cat_names)),
    )


def load_label_map(path: str | Path) -> dict[str, str]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    if not isinstance(loaded, dict):
        raise ValueError("Label map YAML must be an object.")

    candidate: Any
    if "map" in loaded:
        candidate = loaded["map"]
    elif "mapping" in loaded:
        candidate = loaded["mapping"]
    elif "label_map" in loaded:
        candidate = loaded["label_map"]
    else:
        candidate = loaded

    if not isinstance(candidate, dict):
        raise ValueError("Label map must resolve to an object mapping source->target labels.")

    normalized: dict[str, str] = {}
    for src, dst in candidate.items():
        src_name = str(src).strip()
        dst_name = str(dst).strip()
        if not src_name or not dst_name:
            raise ValueError("Label map entries must have non-empty source and target labels.")
        normalized[src_name] = dst_name
    return normalized


def _copy_items(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in items]


def normalize_coco(obj: dict[str, Any], label_map: dict[str, str] | None = None) -> dict[str, Any]:
    """
    Normalize COCO categories and annotation category references.

    Category IDs are rebuilt to be deterministic and contiguous (1..N), while image and
    annotation objects are preserved except for category_id remapping.
    """

    category_map = label_map or {}
    category_id_to_name: dict[int, str] = {}
    merged_names: set[str] = set()

    for idx, category in enumerate(obj["categories"]):
        c = _require_mapping(category, f"categories[{idx}]")
        source_id = _coerce_int(c["id"], f"categories[{idx}].id")
        source_name = str(c["name"]).strip()
        target_name = category_map.get(source_name, source_name)
        category_id_to_name[source_id] = target_name
        merged_names.add(target_name)

    ordered_names = sorted(merged_names)
    name_to_new_id = {name: idx + 1 for idx, name in enumerate(ordered_names)}

    normalized_annotations: list[dict[str, Any]] = []
    for annotation in obj["annotations"]:
        ann = dict(annotation)
        old_category_id = _coerce_int(ann["category_id"], "annotation.category_id")
        new_category_name = category_id_to_name[old_category_id]
        ann["category_id"] = name_to_new_id[new_category_name]
        normalized_annotations.append(ann)

    normalized_categories = [
        {"id": cid, "name": name}
        for name, cid in sorted(name_to_new_id.items(), key=lambda item: item[1])
    ]

    normalized: dict[str, Any] = dict(obj)
    normalized["images"] = _copy_items(obj["images"])
    normalized["annotations"] = normalized_annotations
    normalized["categories"] = normalized_categories
    return normalized


def write_coco(path: str | Path, obj: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return out
