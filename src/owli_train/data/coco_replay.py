from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from owli_train.data.coco import load_coco, validate_coco, write_coco


@dataclass(frozen=True)
class CocoReplayConfig:
    source_coco: Path
    source_images_dir: Path
    label_map_path: Path
    out_dir: Path
    min_bbox_min_side: float
    max_positive_images_per_class: int


@dataclass(frozen=True)
class CocoReplayArtifacts:
    out_dir: Path
    coco_path: Path
    class_names_path: Path
    qc_report_path: Path
    images: int
    annotations: int
    categories: int


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


def _load_label_mapping(path: Path) -> tuple[dict[str, str], list[str], bool]:
    payload = _load_yaml(path)
    candidate = payload.get("map", payload)
    if not isinstance(candidate, dict):
        raise ValueError("Replay label map must contain a `map` object.")

    mapping: dict[str, str] = {}
    for source_name, target_name in candidate.items():
        source = str(source_name).strip()
        target = str(target_name).strip()
        if not source or not target:
            raise ValueError("Replay label map entries must be non-empty.")
        mapping[source] = target

    allowed_target_classes = payload.get("allowed_target_classes", [])
    if not isinstance(allowed_target_classes, list) or not allowed_target_classes:
        raise ValueError("Replay label map must contain a non-empty allowed_target_classes list.")
    contract_order = [str(item).strip() for item in allowed_target_classes]
    if any(not item for item in contract_order):
        raise ValueError("allowed_target_classes must not contain empty values.")

    unknown_targets = sorted(set(mapping.values()) - set(contract_order))
    if unknown_targets:
        preview = ", ".join(unknown_targets[:10])
        suffix = "" if len(unknown_targets) <= 10 else f" (+{len(unknown_targets) - 10} more)"
        raise ValueError(
            "Replay label map targets must stay inside allowed_target_classes: "
            f"{preview}{suffix}"
        )

    return mapping, contract_order, bool(payload.get("drop_unmapped", False))


def load_coco_replay_config(path: Path) -> CocoReplayConfig:
    payload = _load_yaml(path)

    try:
        min_bbox_min_side = float(payload.get("selection", {}).get("min_bbox_min_side", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("selection.min_bbox_min_side must be a number >= 0.") from exc
    if min_bbox_min_side < 0.0:
        raise ValueError("selection.min_bbox_min_side must be >= 0.")

    try:
        max_positive_images_per_class = int(
            payload.get("selection", {}).get("max_positive_images_per_class", 0)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("selection.max_positive_images_per_class must be an integer > 0.") from exc
    if max_positive_images_per_class <= 0:
        raise ValueError("selection.max_positive_images_per_class must be > 0.")

    return CocoReplayConfig(
        source_coco=_resolve_path(payload.get("source_coco"), config_path=path),
        source_images_dir=_resolve_path(
            payload.get("source_images_dir"),
            config_path=path,
            expect_dir=True,
        ),
        label_map_path=_resolve_path(payload.get("label_map"), config_path=path),
        out_dir=_resolve_out_dir(payload.get("out_dir"), config_path=path),
        min_bbox_min_side=min_bbox_min_side,
        max_positive_images_per_class=max_positive_images_per_class,
    )


def _write_class_names(path: Path, class_names: list[str]) -> Path:
    payload = {"class_names": class_names, "category_ids": list(range(1, len(class_names) + 1))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _candidate_rank(
    *,
    image_id: int,
    class_name: str,
    annotations: list[dict[str, Any]],
    category_name_by_id: dict[int, str],
) -> tuple[int, int, int, int]:
    target_hits = sum(
        1 for annotation in annotations if category_name_by_id[int(annotation["category_id"])] == class_name
    )
    distinct_classes = len(
        {category_name_by_id[int(annotation["category_id"])] for annotation in annotations}
    )
    total_annotations = len(annotations)
    return (-target_hits, -distinct_classes, -total_annotations, image_id)


def import_coco_replay_with_config(config_path: Path) -> CocoReplayArtifacts:
    config = load_coco_replay_config(config_path)
    mapping, contract_order, drop_unmapped = _load_label_mapping(config.label_map_path)

    coco = load_coco(config.source_coco)
    _ = validate_coco(coco, images_dir=config.source_images_dir)

    images_by_id = {int(image["id"]): dict(image) for image in coco["images"]}
    source_name_by_category_id = {int(category["id"]): str(category["name"]) for category in coco["categories"]}
    target_id_by_name = {name: idx + 1 for idx, name in enumerate(contract_order)}

    kept_annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    class_counts_before = Counter()
    filtered_small_bbox_annotations = 0
    filtered_unmapped_annotations = 0

    for annotation in coco["annotations"]:
        source_name = source_name_by_category_id.get(int(annotation["category_id"]))
        if source_name is None:
            continue
        target_name = mapping.get(source_name)
        if target_name is None:
            if drop_unmapped:
                filtered_unmapped_annotations += 1
                continue
            target_name = source_name

        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        _, _, width, height = [float(value) for value in bbox]
        if min(width, height) < config.min_bbox_min_side:
            filtered_small_bbox_annotations += 1
            continue

        replay_annotation = dict(annotation)
        replay_annotation["category_id"] = target_id_by_name[target_name]
        image_id = int(replay_annotation["image_id"])
        kept_annotations_by_image.setdefault(image_id, []).append(replay_annotation)
        class_counts_before[target_name] += 1

    positive_image_ids_by_class: dict[str, set[int]] = {name: set() for name in contract_order}
    for image_id, image_annotations in kept_annotations_by_image.items():
        present_classes = {
            contract_order[int(annotation["category_id"]) - 1] for annotation in image_annotations
        }
        for class_name in present_classes:
            positive_image_ids_by_class[class_name].add(image_id)

    class_selection_order = sorted(
        contract_order,
        key=lambda name: (len(positive_image_ids_by_class[name]), contract_order.index(name)),
    )

    selected_image_ids: set[int] = set()
    selected_image_presence = Counter()
    for class_name in class_selection_order:
        candidates = sorted(
            positive_image_ids_by_class[class_name],
            key=lambda image_id: _candidate_rank(
                image_id=image_id,
                class_name=class_name,
                annotations=kept_annotations_by_image[image_id],
                category_name_by_id={idx + 1: name for idx, name in enumerate(contract_order)},
            ),
        )
        for image_id in candidates:
            if selected_image_presence[class_name] >= config.max_positive_images_per_class:
                break
            if image_id in selected_image_ids:
                continue
            selected_image_ids.add(image_id)
            present_classes = {
                contract_order[int(annotation["category_id"]) - 1]
                for annotation in kept_annotations_by_image[image_id]
            }
            for present_class in present_classes:
                selected_image_presence[present_class] += 1

    selected_images = [
        images_by_id[image_id]
        for image_id in sorted(selected_image_ids)
        if image_id in kept_annotations_by_image
    ]
    selected_annotations: list[dict[str, Any]] = []
    selected_annotation_counts = Counter()
    for image in selected_images:
        for annotation in kept_annotations_by_image[int(image["id"])]:
            selected_annotations.append(annotation)
            selected_annotation_counts[contract_order[int(annotation["category_id"]) - 1]] += 1

    selected_categories = [
        {"id": target_id_by_name[name], "name": name}
        for name in contract_order
        if selected_annotation_counts.get(name, 0) > 0
    ]

    out_coco: dict[str, Any] = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": selected_categories,
    }
    for extra_key in ("info", "licenses"):
        if extra_key in coco:
            out_coco[extra_key] = coco[extra_key]

    config.out_dir.mkdir(parents=True, exist_ok=True)
    coco_path = write_coco(config.out_dir / "instances_ba_v1.coco.json", out_coco)
    class_names = [str(category["name"]) for category in selected_categories]
    class_names_path = _write_class_names(config.out_dir / "class_names.json", class_names)

    qc_report = {
        "config_path": str(config_path),
        "source_coco": str(config.source_coco),
        "source_images_dir": str(config.source_images_dir),
        "label_map": str(config.label_map_path),
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
        "contract_order": contract_order,
        "class_selection_order": class_selection_order,
        "filtered_unmapped_annotations": filtered_unmapped_annotations,
        "filtered_small_bbox_annotations": filtered_small_bbox_annotations,
        "dropped_images_without_kept_annotations": len(coco["images"]) - len(kept_annotations_by_image),
        "class_counts_before": {name: class_counts_before[name] for name in contract_order if class_counts_before[name] > 0},
        "selected_annotation_counts": {
            name: selected_annotation_counts[name] for name in contract_order if selected_annotation_counts[name] > 0
        },
        "selected_image_counts": {
            name: selected_image_presence[name] for name in contract_order if selected_image_presence[name] > 0
        },
    }
    qc_report_path = config.out_dir / "qc_report.json"
    qc_report_path.write_text(json.dumps(qc_report, indent=2), encoding="utf-8")

    _ = validate_coco(out_coco, images_dir=config.source_images_dir)

    return CocoReplayArtifacts(
        out_dir=config.out_dir,
        coco_path=coco_path,
        class_names_path=class_names_path,
        qc_report_path=qc_report_path,
        images=len(selected_images),
        annotations=len(selected_annotations),
        categories=len(selected_categories),
    )
