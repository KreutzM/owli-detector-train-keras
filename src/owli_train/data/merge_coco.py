from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from owli_train.data.coco import (
    load_coco,
    load_label_contract_class_names,
    load_label_map,
    normalize_coco,
    validate_coco,
)


class CocoMergeError(RuntimeError):
    """Raised when a COCO merge manifest is invalid or merge fails."""


@dataclass(frozen=True)
class MergeSource:
    name: str
    coco_path: Path
    images_dir: Path | None
    label_map_path: Path | None
    contract_path: Path | None
    pseudo: bool
    score_threshold: float
    image_namespace: str | None
    file_name_prefix: str | None


@dataclass(frozen=True)
class MergeSettings:
    same_class_iou: float = 0.75
    pseudo_block_iou: float = 0.6
    allow_duplicate_file_names: bool = False


@dataclass(frozen=True)
class MergeManifest:
    path: Path
    sources: list[MergeSource]
    settings: MergeSettings


@dataclass(frozen=True)
class MergeCocoArtifacts:
    coco_path: Path
    report_path: Path
    images: int
    annotations: int
    categories: int


def _as_mapping(obj: Any, label: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise CocoMergeError(f"{label} must be a JSON/YAML object.")
    return obj


def _as_list(obj: Any, label: str) -> list[Any]:
    if not isinstance(obj, list):
        raise CocoMergeError(f"{label} must be a list.")
    return obj


def _resolve_path(value: str | None, *, base_dir: Path, label: str) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    raw = Path(text)
    resolved = raw if raw.is_absolute() else (base_dir / raw)
    if label.endswith("_dir"):
        if not resolved.is_dir():
            raise CocoMergeError(f"{label} directory was not found: {resolved}")
    else:
        if not resolved.is_file():
            raise CocoMergeError(f"{label} file was not found: {resolved}")
    return resolved


def _coerce_prob(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CocoMergeError(f"{label} must be a number in [0.0, 1.0].") from exc
    if number < 0.0 or number > 1.0:
        raise CocoMergeError(f"{label} must be in [0.0, 1.0].")
    return number


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def load_merge_manifest(path: Path) -> MergeManifest:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = _as_mapping(payload, str(path))
    base_dir = path.parent

    raw_sources = _as_list(root.get("sources"), "sources")
    if not raw_sources:
        raise CocoMergeError("sources must contain at least one source entry.")

    settings_raw = _as_mapping(root.get("settings", {}), "settings")
    settings = MergeSettings(
        same_class_iou=_coerce_prob(settings_raw.get("same_class_iou", 0.75), "settings.same_class_iou"),
        pseudo_block_iou=_coerce_prob(
            settings_raw.get("pseudo_block_iou", 0.6), "settings.pseudo_block_iou"
        ),
        allow_duplicate_file_names=_coerce_bool(
            settings_raw.get("allow_duplicate_file_names"),
            default=False,
        ),
    )

    sources: list[MergeSource] = []
    seen_names: set[str] = set()

    for idx, entry in enumerate(raw_sources):
        item = _as_mapping(entry, f"sources[{idx}]")
        name = str(item.get("name") or "").strip() or f"source_{idx + 1}"
        if name in seen_names:
            raise CocoMergeError(f"Duplicate source name in manifest: {name}")
        seen_names.add(name)

        coco_path = _resolve_path(
            str(item.get("coco", "")).strip() or None,
            base_dir=base_dir,
            label=f"sources[{idx}].coco",
        )
        if coco_path is None:
            raise CocoMergeError(f"sources[{idx}].coco must be set.")

        images_dir = _resolve_path(
            str(item.get("images_dir", "")).strip() or None,
            base_dir=base_dir,
            label=f"sources[{idx}].images_dir",
        )

        label_map_path = _resolve_path(
            str(item.get("label_map", "")).strip() or None,
            base_dir=base_dir,
            label=f"sources[{idx}].label_map",
        )
        contract_path = _resolve_path(
            str(item.get("contract", "")).strip() or None,
            base_dir=base_dir,
            label=f"sources[{idx}].contract",
        )

        pseudo = bool(item.get("pseudo", False))
        score_threshold = _coerce_prob(
            item.get("score_threshold", 0.0),
            f"sources[{idx}].score_threshold",
        )
        image_namespace = str(item.get("image_namespace") or "").strip() or None
        file_name_prefix = str(item.get("file_name_prefix") or "").strip().strip("/") or None

        sources.append(
            MergeSource(
                name=name,
                coco_path=coco_path,
                images_dir=images_dir,
                label_map_path=label_map_path,
                contract_path=contract_path,
                pseudo=pseudo,
                score_threshold=score_threshold,
                image_namespace=image_namespace,
                file_name_prefix=file_name_prefix,
            )
        )

    return MergeManifest(path=path, sources=sources, settings=settings)


def _bbox_iou_xywh(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx0, ly0, lw, lh = left
    rx0, ry0, rw, rh = right
    lx1, ly1 = lx0 + lw, ly0 + lh
    rx1, ry1 = rx0 + rw, ry0 + rh

    ix0 = max(lx0, rx0)
    iy0 = max(ly0, ry0)
    ix1 = min(lx1, rx1)
    iy1 = min(ly1, ry1)

    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    left_area = max(0.0, lw) * max(0.0, lh)
    right_area = max(0.0, rw) * max(0.0, rh)
    denom = left_area + right_area - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def _prefixed_file_name(file_name: str, prefix: str | None) -> str:
    if prefix is None:
        return file_name
    return f"{prefix}/{file_name}".replace("//", "/")


def _source_namespace(source: MergeSource) -> str:
    if source.image_namespace:
        return source.image_namespace
    if source.images_dir is not None:
        return str(source.images_dir.resolve())
    return source.name


def merge_coco_from_manifest(
    *,
    manifest_path: Path,
    out_path: Path,
    report_out_path: Path | None = None,
) -> MergeCocoArtifacts:
    manifest = load_merge_manifest(manifest_path)
    settings = manifest.settings

    global_categories: list[dict[str, Any]] = []
    global_category_by_name: dict[str, int] = {}

    global_images: list[dict[str, Any]] = []
    next_image_id = 1
    image_key_to_global_id: dict[tuple[str, str], int] = {}
    image_id_to_entry: dict[int, dict[str, Any]] = {}

    seen_file_name_namespace: dict[str, str] = {}

    kept_annotations: list[dict[str, Any]] = []
    kept_active: list[bool] = []
    kept_by_image: dict[int, list[int]] = {}

    drops = {
        "duplicate_gt_same_class": 0,
        "pseudo_low_score": 0,
        "pseudo_overlap_gt": 0,
        "pseudo_duplicate_same_class": 0,
        "pseudo_replaced_by_gt": 0,
        "invalid_bbox": 0,
    }

    source_reports: list[dict[str, Any]] = []

    for source_index, source in enumerate(manifest.sources):
        source_coco = load_coco(source.coco_path)
        validate_coco(source_coco, images_dir=source.images_dir)

        label_map = load_label_map(source.label_map_path) if source.label_map_path is not None else None
        category_order = (
            load_label_contract_class_names(source.contract_path)
            if source.contract_path is not None
            else None
        )
        normalized = normalize_coco(source_coco, label_map=label_map, category_order=category_order)

        categories = sorted(normalized["categories"], key=lambda cat: int(cat["id"]))
        category_by_id = {int(cat["id"]): str(cat["name"]) for cat in categories}

        local_to_global_category: dict[int, int] = {}
        for local_id, category_name in category_by_id.items():
            if category_name not in global_category_by_name:
                new_id = len(global_categories) + 1
                global_category_by_name[category_name] = new_id
                global_categories.append({"id": new_id, "name": category_name})
            local_to_global_category[local_id] = global_category_by_name[category_name]

        namespace = _source_namespace(source)
        image_id_remap: dict[int, int] = {}

        for image in normalized["images"]:
            local_image_id = int(image["id"])
            file_name = _prefixed_file_name(str(image["file_name"]), source.file_name_prefix)

            owner_namespace = seen_file_name_namespace.get(file_name)
            if (
                owner_namespace is not None
                and owner_namespace != namespace
                and not settings.allow_duplicate_file_names
            ):
                raise CocoMergeError(
                    "Detected duplicate file_name across different source namespaces: "
                    f"{file_name}. Use sources[].file_name_prefix or set "
                    "settings.allow_duplicate_file_names=true intentionally."
                )
            seen_file_name_namespace[file_name] = namespace

            key = (namespace, file_name)
            if key in image_key_to_global_id:
                global_image_id = image_key_to_global_id[key]
                existing = image_id_to_entry[global_image_id]
                width = int(image.get("width") or 0)
                height = int(image.get("height") or 0)
                if int(existing.get("width") or 0) != width or int(existing.get("height") or 0) != height:
                    raise CocoMergeError(
                        "Image dimension mismatch for shared image key "
                        f"{file_name} in namespace {namespace}."
                    )
            else:
                global_image_id = next_image_id
                next_image_id += 1
                out_image = {
                    "id": global_image_id,
                    "file_name": file_name,
                    "width": int(image.get("width") or 0),
                    "height": int(image.get("height") or 0),
                }
                global_images.append(out_image)
                image_id_to_entry[global_image_id] = out_image
                image_key_to_global_id[key] = global_image_id

            image_id_remap[local_image_id] = global_image_id

        source_kept = 0
        source_input_annotations = 0
        for ann in sorted(normalized["annotations"], key=lambda item: int(item["id"])):
            source_input_annotations += 1
            local_image_id = int(ann["image_id"])
            local_category_id = int(ann["category_id"])
            if local_image_id not in image_id_remap:
                continue
            if local_category_id not in local_to_global_category:
                continue

            score = float(ann.get("score", 1.0))
            if source.pseudo and score < source.score_threshold:
                drops["pseudo_low_score"] += 1
                continue

            bbox_raw = ann.get("bbox") or []
            if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
                drops["invalid_bbox"] += 1
                continue
            bbox = tuple(float(v) for v in bbox_raw)
            if bbox[2] <= 0.0 or bbox[3] <= 0.0:
                drops["invalid_bbox"] += 1
                continue

            global_image_id = image_id_remap[local_image_id]
            global_category_id = local_to_global_category[local_category_id]

            candidate = {
                "image_id": global_image_id,
                "category_id": global_category_id,
                "bbox": [float(v) for v in bbox],
                "area": float(max(0.0, bbox[2]) * max(0.0, bbox[3])),
                "iscrowd": int(ann.get("iscrowd", 0)),
                "source_name": source.name,
                "source_index": source_index,
                "is_pseudo": source.pseudo,
                "score": float(score),
            }

            if source.pseudo:
                skip_reason = None
                for kept_index in kept_by_image.get(global_image_id, []):
                    if not kept_active[kept_index]:
                        continue
                    kept = kept_annotations[kept_index]
                    iou = _bbox_iou_xywh(tuple(candidate["bbox"]), tuple(kept["bbox"]))
                    if not kept["is_pseudo"] and iou >= settings.pseudo_block_iou:
                        skip_reason = "pseudo_overlap_gt"
                        break
                    if kept["category_id"] == global_category_id and iou >= settings.same_class_iou:
                        skip_reason = "pseudo_duplicate_same_class"
                        break
                if skip_reason is not None:
                    drops[skip_reason] += 1
                    continue
            else:
                duplicate_gt = False
                pseudo_to_disable: list[int] = []
                for kept_index in kept_by_image.get(global_image_id, []):
                    if not kept_active[kept_index]:
                        continue
                    kept = kept_annotations[kept_index]
                    if kept["category_id"] != global_category_id:
                        continue
                    iou = _bbox_iou_xywh(tuple(candidate["bbox"]), tuple(kept["bbox"]))
                    if iou < settings.same_class_iou:
                        continue
                    if kept["is_pseudo"]:
                        pseudo_to_disable.append(kept_index)
                    else:
                        duplicate_gt = True
                        break

                if duplicate_gt:
                    drops["duplicate_gt_same_class"] += 1
                    continue

                for kept_index in pseudo_to_disable:
                    if kept_active[kept_index]:
                        kept_active[kept_index] = False
                        drops["pseudo_replaced_by_gt"] += 1

            kept_index = len(kept_annotations)
            kept_annotations.append(candidate)
            kept_active.append(True)
            kept_by_image.setdefault(global_image_id, []).append(kept_index)
            source_kept += 1

        source_reports.append(
            {
                "name": source.name,
                "coco": str(source.coco_path),
                "images_dir": str(source.images_dir) if source.images_dir is not None else None,
                "pseudo": bool(source.pseudo),
                "score_threshold": float(source.score_threshold),
                "images_in_source": len(normalized["images"]),
                "annotations_in_source": source_input_annotations,
                "annotations_kept": source_kept,
            }
        )

    final_annotations: list[dict[str, Any]] = []
    next_ann_id = 1
    for index, ann in enumerate(kept_annotations):
        if not kept_active[index]:
            continue
        out_ann = {
            "id": next_ann_id,
            "image_id": int(ann["image_id"]),
            "category_id": int(ann["category_id"]),
            "bbox": [float(v) for v in ann["bbox"]],
            "area": float(ann["area"]),
            "iscrowd": int(ann["iscrowd"]),
        }
        if ann["is_pseudo"]:
            out_ann["score"] = float(ann["score"])
        final_annotations.append(out_ann)
        next_ann_id += 1

    merged = {
        "images": sorted(global_images, key=lambda item: int(item["id"])),
        "annotations": final_annotations,
        "categories": sorted(global_categories, key=lambda item: int(item["id"])),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    report_path = report_out_path if report_out_path is not None else out_path.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "manifest_path": str(manifest.path),
        "out_path": str(out_path),
        "settings": {
            "same_class_iou": float(settings.same_class_iou),
            "pseudo_block_iou": float(settings.pseudo_block_iou),
            "allow_duplicate_file_names": bool(settings.allow_duplicate_file_names),
        },
        "summary": {
            "images": len(merged["images"]),
            "annotations": len(merged["annotations"]),
            "categories": len(merged["categories"]),
        },
        "drops": {key: int(value) for key, value in drops.items()},
        "categories": [str(item["name"]) for item in merged["categories"]],
        "sources": source_reports,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return MergeCocoArtifacts(
        coco_path=out_path,
        report_path=report_path,
        images=len(merged["images"]),
        annotations=len(merged["annotations"]),
        categories=len(merged["categories"]),
    )
