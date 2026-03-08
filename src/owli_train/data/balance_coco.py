from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from owli_train.data.coco import load_coco, validate_coco, write_coco
from owli_train.data.split import write_splits


@dataclass(frozen=True)
class BalanceCocoConfig:
    source_coco: Path
    source_images_dir: Path | None
    source_splits_json: Path | None
    out_dir: Path
    min_bbox_min_side: float
    max_positive_images_per_class: int


@dataclass(frozen=True)
class BalanceCocoArtifacts:
    out_dir: Path
    coco_path: Path
    class_names_path: Path
    qc_report_path: Path
    splits_path: Path | None
    images: int
    annotations: int
    categories: int


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain an object: {path}")
    return payload


def _resolve_optional_path(
    value: Any,
    *,
    config_path: Path,
    expect_dir: bool = False,
) -> Path | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    path = Path(raw)
    resolved = path if path.is_absolute() else (config_path.parent / path).resolve()
    if expect_dir:
        if not resolved.is_dir():
            raise ValueError(f"Directory does not exist: {resolved}")
    else:
        if not resolved.is_file():
            raise ValueError(f"File does not exist: {resolved}")
    return resolved


def load_balance_coco_config(path: Path) -> BalanceCocoConfig:
    payload = _load_yaml(path)
    source_coco = _resolve_optional_path(payload.get("source_coco"), config_path=path)
    if source_coco is None:
        raise ValueError("balance config must set source_coco.")

    source_images_dir = _resolve_optional_path(
        payload.get("source_images_dir"),
        config_path=path,
        expect_dir=True,
    )
    source_splits_json = _resolve_optional_path(payload.get("source_splits_json"), config_path=path)

    out_dir_raw = str(payload.get("out_dir") or "").strip()
    if not out_dir_raw:
        raise ValueError("balance config must set out_dir.")
    out_dir_path = Path(out_dir_raw)
    out_dir = out_dir_path if out_dir_path.is_absolute() else (path.parent / out_dir_path).resolve()

    selection = payload.get("selection", {})
    if not isinstance(selection, dict):
        raise ValueError("balance config selection must be an object.")

    try:
        min_bbox_min_side = float(selection.get("min_bbox_min_side", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("selection.min_bbox_min_side must be a number >= 0.") from exc
    if min_bbox_min_side < 0.0:
        raise ValueError("selection.min_bbox_min_side must be >= 0.")

    try:
        max_positive_images_per_class = int(selection.get("max_positive_images_per_class", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("selection.max_positive_images_per_class must be an integer > 0.") from exc
    if max_positive_images_per_class <= 0:
        raise ValueError("selection.max_positive_images_per_class must be > 0.")

    return BalanceCocoConfig(
        source_coco=source_coco,
        source_images_dir=source_images_dir,
        source_splits_json=source_splits_json,
        out_dir=out_dir,
        min_bbox_min_side=min_bbox_min_side,
        max_positive_images_per_class=max_positive_images_per_class,
    )


def _load_splits(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("source_splits_json must contain a JSON object.")

    assignments: dict[int, str] = {}
    for split_name in ("train", "val", "test"):
        values = raw.get(split_name, [])
        if not isinstance(values, list):
            raise ValueError(f"source_splits_json key `{split_name}` must be a list.")
        for image_id in values:
            assignments[int(image_id)] = split_name
    return assignments


def _write_class_names(path: Path, categories: list[dict[str, Any]]) -> Path:
    ordered = sorted(categories, key=lambda cat: int(cat["id"]))
    payload = {
        "class_names": [str(cat["name"]) for cat in ordered],
        "category_ids": [int(cat["id"]) for cat in ordered],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def balance_coco_with_config(config_path: Path) -> BalanceCocoArtifacts:
    config = load_balance_coco_config(config_path)
    coco = load_coco(config.source_coco)
    _ = validate_coco(coco, images_dir=config.source_images_dir)

    category_name_by_id = {int(cat["id"]): str(cat["name"]) for cat in coco["categories"]}
    ordered_category_names = [str(cat["name"]) for cat in sorted(coco["categories"], key=lambda cat: int(cat["id"]))]
    images_by_id = {int(image["id"]): dict(image) for image in coco["images"]}
    split_by_image_id = _load_splits(config.source_splits_json)

    kept_annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    filtered_small_bbox_annotations = 0
    for annotation in coco["annotations"]:
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        _, _, width, height = [float(value) for value in bbox]
        if min(width, height) < config.min_bbox_min_side:
            filtered_small_bbox_annotations += 1
            continue
        image_id = int(annotation["image_id"])
        kept_annotations_by_image.setdefault(image_id, []).append(dict(annotation))

    kept_annotations_by_image = {
        image_id: annotations for image_id, annotations in kept_annotations_by_image.items() if annotations
    }

    image_ids_by_class: dict[str, set[int]] = {name: set() for name in ordered_category_names}
    class_counts_before: dict[str, int] = {name: 0 for name in ordered_category_names}
    for image_id, annotations in kept_annotations_by_image.items():
        present_classes: set[str] = set()
        for annotation in annotations:
            class_name = category_name_by_id[int(annotation["category_id"])]
            class_counts_before[class_name] += 1
            present_classes.add(class_name)
        for class_name in present_classes:
            image_ids_by_class[class_name].add(image_id)

    class_selection_order = sorted(
        ordered_category_names,
        key=lambda name: (len(image_ids_by_class[name]), ordered_category_names.index(name)),
    )

    selected_image_ids: set[int] = set()
    selected_image_counts_by_class: dict[str, int] = {name: 0 for name in ordered_category_names}

    def _candidate_rank(image_id: int, class_name: str) -> tuple[int, int, int, int]:
        annotations = kept_annotations_by_image[image_id]
        target_hits = sum(
            1 for annotation in annotations if category_name_by_id[int(annotation["category_id"])] == class_name
        )
        distinct_classes = len({category_name_by_id[int(annotation["category_id"])] for annotation in annotations})
        total_annotations = len(annotations)
        return (-target_hits, -distinct_classes, -total_annotations, image_id)

    for class_name in class_selection_order:
        candidates = sorted(image_ids_by_class[class_name], key=lambda image_id: _candidate_rank(image_id, class_name))
        for image_id in candidates:
            if selected_image_counts_by_class[class_name] >= config.max_positive_images_per_class:
                break
            if image_id in selected_image_ids:
                continue
            selected_image_ids.add(image_id)
            present_classes = {
                category_name_by_id[int(annotation["category_id"])]
                for annotation in kept_annotations_by_image[image_id]
            }
            for present_class in present_classes:
                selected_image_counts_by_class[present_class] += 1

    selected_images = sorted(
        (images_by_id[image_id] for image_id in selected_image_ids),
        key=lambda image: int(image["id"]),
    )
    selected_annotations: list[dict[str, Any]] = []
    selected_annotation_counts: dict[str, int] = {name: 0 for name in ordered_category_names}
    selected_image_presence: dict[str, int] = {name: 0 for name in ordered_category_names}
    selected_splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for image in selected_images:
        image_id = int(image["id"])
        present_classes: set[str] = set()
        for annotation in kept_annotations_by_image[image_id]:
            selected_annotations.append(annotation)
            class_name = category_name_by_id[int(annotation["category_id"])]
            selected_annotation_counts[class_name] += 1
            present_classes.add(class_name)
        for class_name in present_classes:
            selected_image_presence[class_name] += 1
        split_name = split_by_image_id.get(image_id)
        if split_name in selected_splits:
            selected_splits[split_name].append(image_id)

    present_category_ids = {int(annotation["category_id"]) for annotation in selected_annotations}
    selected_categories = [
        dict(category)
        for category in sorted(coco["categories"], key=lambda cat: int(cat["id"]))
        if int(category["id"]) in present_category_ids
    ]

    out_coco = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": selected_categories,
    }
    for extra_key in ("info", "licenses"):
        if extra_key in coco:
            out_coco[extra_key] = coco[extra_key]

    config.out_dir.mkdir(parents=True, exist_ok=True)
    coco_path = write_coco(config.out_dir / "instances_ba_v1.coco.json", out_coco)
    class_names_path = _write_class_names(config.out_dir / "class_names.json", selected_categories)

    splits_path: Path | None = None
    if any(selected_splits[split_name] for split_name in ("train", "val", "test")):
        splits_path = write_splits(config.out_dir, selected_splits)

    qc_report = {
        "config_path": str(config_path),
        "source_coco": str(config.source_coco),
        "source_images_dir": str(config.source_images_dir) if config.source_images_dir is not None else None,
        "source_splits_json": str(config.source_splits_json) if config.source_splits_json is not None else None,
        "selection": {
            "min_bbox_min_side": config.min_bbox_min_side,
            "max_positive_images_per_class": config.max_positive_images_per_class,
        },
        "input": {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
            "categories": len(coco["categories"]),
        },
        "output": {
            "images": len(selected_images),
            "annotations": len(selected_annotations),
            "categories": len(selected_categories),
        },
        "class_selection_order": class_selection_order,
        "filtered_small_bbox_annotations": filtered_small_bbox_annotations,
        "class_counts_before": {name: count for name, count in class_counts_before.items() if count > 0},
        "selected_annotation_counts": {
            name: count for name, count in selected_annotation_counts.items() if count > 0
        },
        "selected_image_counts": {
            name: count for name, count in selected_image_presence.items() if count > 0
        },
        "selected_split_counts": {
            split_name: len(image_ids)
            for split_name, image_ids in selected_splits.items()
            if image_ids
        },
    }
    qc_report_path = config.out_dir / "qc_report.json"
    qc_report_path.write_text(json.dumps(qc_report, indent=2), encoding="utf-8")

    _ = validate_coco(out_coco, images_dir=config.source_images_dir)

    return BalanceCocoArtifacts(
        out_dir=config.out_dir,
        coco_path=coco_path,
        class_names_path=class_names_path,
        qc_report_path=qc_report_path,
        splits_path=splits_path,
        images=len(selected_images),
        annotations=len(selected_annotations),
        categories=len(selected_categories),
    )
