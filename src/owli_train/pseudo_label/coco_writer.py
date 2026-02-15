from __future__ import annotations

import json
from collections import Counter
from importlib import resources
from pathlib import Path
from typing import Any


def load_coco80_categories() -> list[dict[str, Any]]:
    path = resources.files("owli_train.assets").joinpath("coco80_categories.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Invalid coco80_categories.json payload.")
    categories: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Category entry must be an object.")
        category_id = int(item["id"])
        name = str(item["name"])
        categories.append({"id": category_id, "name": name})
    return categories


def parse_classes_filter(
    raw: str | None,
    *,
    categories: list[dict[str, Any]],
) -> set[int] | None:
    if raw is None or not raw.strip():
        return None

    category_ids = [int(item["id"]) for item in categories]
    by_name = {str(item["name"]).strip().lower(): int(item["id"]) for item in categories}
    by_id = set(category_ids)
    selected: set[int] = set()

    for token in (part.strip() for part in raw.split(",")):
        if not token:
            continue
        lower = token.lower()
        if lower in by_name:
            selected.add(by_name[lower])
            continue
        if token.isdigit():
            value = int(token)
            if value in by_id:
                selected.add(value)
                continue
            if 0 <= value < len(category_ids):
                selected.add(category_ids[value])
                continue
            if 1 <= value <= len(category_ids):
                selected.add(category_ids[value - 1])
                continue
        raise ValueError(f"Unknown class filter token: {token}")

    if not selected:
        raise ValueError("Class filter resolved to no categories.")
    return selected


def build_pseudo_coco(
    *,
    images: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    categories: list[dict[str, Any]],
) -> dict[str, Any]:
    annotations: list[dict[str, Any]] = []
    next_id = 1
    for det in detections:
        bbox = [float(v) for v in det["bbox"]]
        area = float(max(0.0, bbox[2]) * max(0.0, bbox[3]))
        annotations.append(
            {
                "id": next_id,
                "image_id": int(det["image_id"]),
                "category_id": int(det["category_id"]),
                "bbox": bbox,
                "score": float(det["score"]),
                "area": area,
                "iscrowd": 0,
            }
        )
        next_id += 1

    coco = {
        "images": [
            {
                "id": int(item["id"]),
                "file_name": str(item["file_name"]),
                "width": int(item["width"]),
                "height": int(item["height"]),
            }
            for item in images
        ],
        "annotations": annotations,
        "categories": [{"id": int(item["id"]), "name": str(item["name"])} for item in categories],
    }
    return coco


def _score_histogram(scores: list[float]) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    for idx in range(10):
        start = idx / 10.0
        end = (idx + 1) / 10.0
        if idx == 9:
            count = sum(1 for score in scores if start <= score <= end)
        else:
            count = sum(1 for score in scores if start <= score < end)
        bins.append({"min": round(start, 2), "max": round(end, 2), "count": count})
    return bins


def build_pseudo_report(
    *,
    num_images: int,
    detections: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    total_seconds: float,
    teacher_source: str,
    batch_size: int,
    input_size: int,
    score_threshold: float,
    max_detections_per_image: int,
) -> dict[str, Any]:
    category_names = {int(item["id"]): str(item["name"]) for item in categories}
    by_category = Counter(int(item["category_id"]) for item in detections)
    scores = [float(item["score"]) for item in detections]
    total_detections = len(detections)
    images_per_sec = float(num_images / total_seconds) if total_seconds > 0 else 0.0
    avg_det_per_image = float(total_detections / num_images) if num_images > 0 else 0.0

    return {
        "num_images": int(num_images),
        "total_detections": int(total_detections),
        "average_detections_per_image": avg_det_per_image,
        "per_class_counts": {
            category_names[cat_id]: int(count)
            for cat_id, count in sorted(by_category.items(), key=lambda item: item[0])
            if cat_id in category_names
        },
        "score_histogram": _score_histogram(scores),
        "runtime": {
            "total_seconds": float(total_seconds),
            "images_per_second": images_per_sec,
            "batch_size": int(batch_size),
        },
        "settings": {
            "teacher_source": teacher_source,
            "input_size": int(input_size),
            "score_threshold": float(score_threshold),
            "max_detections_per_image": int(max_detections_per_image),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
