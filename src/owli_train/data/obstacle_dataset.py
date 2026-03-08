from __future__ import annotations

import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from owli_train.data.coco import validate_coco, write_coco
from owli_train.data.split import build_split_coco, write_splits


@dataclass(frozen=True)
class ObstacleDatasetSplitArtifacts:
    scanned_xml_files: int
    images: int
    annotations: int
    skipped_missing_images: int
    skipped_images_without_mapped_annotations: int
    skipped_annotations_invalid_bbox: int
    resolved_duplicate_filenames: int
    skipped_ambiguous_duplicate_filenames: int
    target_class_counts: dict[str, int]
    source_class_counts: dict[str, int]
    unmapped_source_class_counts: dict[str, int]


@dataclass(frozen=True)
class ObstacleDatasetImportArtifacts:
    out_dir: Path
    images_dir: Path
    combined_coco_path: Path
    train_coco_path: Path
    val_coco_path: Path
    test_coco_path: Path
    splits_path: Path
    class_names_path: Path
    qc_report_path: Path
    train: ObstacleDatasetSplitArtifacts
    val: ObstacleDatasetSplitArtifacts
    test: ObstacleDatasetSplitArtifacts
    categories: list[str]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain an object: {path}")
    return payload


def _load_label_mapping(map_path: Path) -> tuple[dict[str, str], list[str], list[str]]:
    payload = _load_yaml(map_path)
    raw_map = payload.get("map", payload)
    if not isinstance(raw_map, dict):
        raise ValueError(
            "Obstacle-Dataset label map must resolve to an object mapping source->target labels."
        )
    mapping = {str(src).strip(): str(dst).strip() for src, dst in raw_map.items()}
    if any(not src or not dst for src, dst in mapping.items()):
        raise ValueError("Obstacle-Dataset label map contains empty source or target labels.")

    allowed_targets = [str(item).strip() for item in payload.get("allowed_target_classes", [])]
    if allowed_targets and not all(allowed_targets):
        raise ValueError("allowed_target_classes must not contain empty values.")

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

    if contract_order and set(mapping.values()) - set(contract_order):
        unknown = sorted(set(mapping.values()) - set(contract_order))
        raise ValueError(
            "Obstacle-Dataset label map targets must stay inside the referenced BA contract: "
            + ", ".join(unknown)
        )

    if allowed_targets and set(mapping.values()) - set(allowed_targets):
        unknown = sorted(set(mapping.values()) - set(allowed_targets))
        raise ValueError(
            "Obstacle-Dataset label map targets must stay inside allowed_target_classes: "
            + ", ".join(unknown)
        )

    return mapping, contract_order, allowed_targets


def _resolve_category_order(mapped_targets: set[str], contract_order: list[str]) -> list[str]:
    if contract_order:
        ordered = [name for name in contract_order if name in mapped_targets]
        if ordered:
            return ordered
    return sorted(mapped_targets)


def _candidate_image_roots(dataset_root: Path, split_name: str) -> list[Path]:
    return [
        dataset_root / f"img-{split_name}",
        dataset_root / "JPEGImages",
        dataset_root / "OD-test" / "JPEGImages",
    ]


def _resolve_image_path(dataset_root: Path, split_name: str, filename: str) -> Path | None:
    for root in _candidate_image_roots(dataset_root, split_name):
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return None


def _write_link_or_copy(*, src: Path, dst: Path, mode: str) -> tuple[int, int]:
    if mode not in {"auto", "symlink", "copy"}:
        raise ValueError("mode must be one of: auto, symlink, copy")

    if mode in {"auto", "symlink"}:
        try:
            os.symlink(src.resolve(), dst)
            return 0, 1
        except OSError as exc:
            if mode == "symlink":
                raise ValueError(f"Failed to create symlink for {dst} -> {src}: {exc}") from exc

    shutil.copy2(src, dst)
    return 1, 0


def _export_image_for_training(*, src: Path, dst: Path, mode: str) -> tuple[int, int]:
    with Image.open(src) as image:
        is_jpeg = image.format == "JPEG"
        if not is_jpeg:
            dst.parent.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").save(dst, format="JPEG", quality=95)
            return 1, 0

    return _write_link_or_copy(src=src, dst=dst, mode=mode)


def _load_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Image has invalid dimensions: {image_path}")
    return width, height


def _parse_voc_bbox(obj: ET.Element, *, width: int, height: int) -> list[float] | None:
    bbox = obj.find("bndbox")
    if bbox is None:
        return None

    try:
        xmin = float((bbox.findtext("xmin") or "").strip())
        ymin = float((bbox.findtext("ymin") or "").strip())
        xmax = float((bbox.findtext("xmax") or "").strip())
        ymax = float((bbox.findtext("ymax") or "").strip())
    except ValueError:
        return None

    xmin = max(0.0, min(float(width), xmin))
    ymin = max(0.0, min(float(height), ymin))
    xmax = max(0.0, min(float(width), xmax))
    ymax = max(0.0, min(float(height), ymax))
    out_w = xmax - xmin
    out_h = ymax - ymin
    if out_w <= 0.0 or out_h <= 0.0:
        return None
    return [round(xmin, 4), round(ymin, 4), round(out_w, 4), round(out_h, 4)]


def _parse_declared_xml_size(root: ET.Element) -> tuple[int, int] | None:
    try:
        width = int(str(root.findtext("./size/width", default="")).strip())
        height = int(str(root.findtext("./size/height", default="")).strip())
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _select_xml_candidate(
    *,
    xml_roots: list[tuple[Path, ET.Element]],
    image_size: tuple[int, int],
) -> tuple[Path, ET.Element] | None:
    matching = [
        (xml_path, root)
        for xml_path, root in xml_roots
        if _parse_declared_xml_size(root) == image_size
    ]
    if len(matching) == 1:
        return matching[0]
    if len(matching) > 1:
        return None
    if len(xml_roots) == 1:
        return xml_roots[0]
    return None


def _build_split_artifacts(
    *,
    scanned_xml_files: int,
    images: int,
    annotations: int,
    skipped_missing_images: int,
    skipped_images_without_mapped_annotations: int,
    skipped_annotations_invalid_bbox: int,
    resolved_duplicate_filenames: int,
    skipped_ambiguous_duplicate_filenames: int,
    target_class_counts: Counter[str],
    source_class_counts: Counter[str],
    unmapped_source_class_counts: Counter[str],
) -> ObstacleDatasetSplitArtifacts:
    return ObstacleDatasetSplitArtifacts(
        scanned_xml_files=scanned_xml_files,
        images=images,
        annotations=annotations,
        skipped_missing_images=skipped_missing_images,
        skipped_images_without_mapped_annotations=skipped_images_without_mapped_annotations,
        skipped_annotations_invalid_bbox=skipped_annotations_invalid_bbox,
        resolved_duplicate_filenames=resolved_duplicate_filenames,
        skipped_ambiguous_duplicate_filenames=skipped_ambiguous_duplicate_filenames,
        target_class_counts=dict(sorted(target_class_counts.items())),
        source_class_counts=dict(sorted(source_class_counts.items())),
        unmapped_source_class_counts=dict(sorted(unmapped_source_class_counts.items())),
    )


def import_obstacle_dataset_to_coco(
    *,
    dataset_dir: Path,
    out_dir: Path,
    label_map_path: Path,
    mode: str = "auto",
) -> ObstacleDatasetImportArtifacts:
    dataset_root = Path(dataset_dir)
    if not dataset_root.is_dir():
        raise ValueError(f"--dataset-dir was not found: {dataset_root}")

    mapping, contract_order, _ = _load_label_mapping(label_map_path)
    if not mapping:
        raise ValueError("Obstacle-Dataset label map is empty; local source taxonomy must be mapped first.")

    split_xml_dirs = {
        "train": dataset_root / "ann-train",
        "val": dataset_root / "ann-val",
        "test": dataset_root / "ann-test",
    }
    for split_name, xml_dir in split_xml_dirs.items():
        if not xml_dir.is_dir():
            raise ValueError(f"Missing Obstacle-Dataset split annotations directory: {xml_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_images_root = out_dir / "images"
    out_images_root.mkdir(parents=True, exist_ok=True)

    image_entries: list[dict[str, Any]] = []
    raw_annotation_entries: list[dict[str, Any]] = []
    split_image_ids: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    used_targets: set[str] = set()
    seen_file_names: set[str] = set()
    split_qc: dict[str, ObstacleDatasetSplitArtifacts] = {}
    missing_image_samples: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    next_image_id = 1
    next_annotation_id = 1
    copied = 0
    symlinked = 0

    for split_name in ("train", "val", "test"):
        target_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        unmapped_source_counts: Counter[str] = Counter()
        scanned_xml_files = 0
        images = 0
        annotations = 0
        skipped_missing_images = 0
        skipped_images_without_mapped_annotations = 0
        skipped_annotations_invalid_bbox = 0
        resolved_duplicate_filenames = 0
        skipped_ambiguous_duplicate_filenames = 0

        xml_groups: dict[str, list[tuple[Path, ET.Element]]] = {}
        for xml_path in sorted(split_xml_dirs[split_name].glob("*.xml")):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            filename = str(root.findtext("./filename", default="")).strip()
            if not filename:
                raise ValueError(f"Missing filename in XML: {xml_path}")
            xml_groups.setdefault(filename, []).append((xml_path, root))

        for filename in sorted(xml_groups):
            xml_candidates = xml_groups[filename]
            scanned_xml_files += len(xml_candidates)
            source_image_path = _resolve_image_path(dataset_root, split_name, filename)
            if source_image_path is None:
                skipped_missing_images += len(xml_candidates)
                if len(missing_image_samples[split_name]) < 25:
                    missing_image_samples[split_name].append(filename)
                continue

            width, height = _load_image_size(source_image_path)
            selected = _select_xml_candidate(
                xml_roots=xml_candidates,
                image_size=(width, height),
            )
            if selected is None:
                skipped_ambiguous_duplicate_filenames += len(xml_candidates)
                continue
            if len(xml_candidates) > 1:
                resolved_duplicate_filenames += 1
            xml_path, root = selected
            mapped_annotations: list[dict[str, Any]] = []

            for obj in root.findall("./object"):
                source_name = str(obj.findtext("name", default="")).strip()
                if not source_name:
                    continue
                source_counts[source_name] += 1
                target_name = mapping.get(source_name)
                if target_name is None:
                    unmapped_source_counts[source_name] += 1
                    continue
                coco_bbox = _parse_voc_bbox(obj, width=width, height=height)
                if coco_bbox is None:
                    skipped_annotations_invalid_bbox += 1
                    continue
                mapped_annotations.append(
                    {
                        "id": next_annotation_id,
                        "category_name": target_name,
                        "bbox": coco_bbox,
                        "area": round(coco_bbox[2] * coco_bbox[3], 4),
                        "iscrowd": 0,
                    }
                )
                next_annotation_id += 1
                target_counts[target_name] += 1
                used_targets.add(target_name)

            if not mapped_annotations:
                skipped_images_without_mapped_annotations += 1
                continue

            file_name = f"{split_name}/{filename}"
            if file_name in seen_file_names:
                raise ValueError(f"Duplicate exported file_name resolved for Obstacle-Dataset import: {file_name}")
            seen_file_names.add(file_name)

            dest_path = out_images_root / file_name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if dest_path.exists() or dest_path.is_symlink():
                dest_path.unlink()
            copied_delta, symlink_delta = _export_image_for_training(
                src=source_image_path,
                dst=dest_path,
                mode=mode,
            )
            copied += copied_delta
            symlinked += symlink_delta

            image_id = next_image_id
            next_image_id += 1
            image_entries.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                }
            )
            split_image_ids[split_name].append(image_id)
            images += 1

            for annotation in mapped_annotations:
                raw_annotation_entries.append(
                    {
                        "id": annotation["id"],
                        "image_id": image_id,
                        "category_name": annotation["category_name"],
                        "bbox": annotation["bbox"],
                        "area": annotation["area"],
                        "iscrowd": annotation["iscrowd"],
                    }
                )
                annotations += 1

        split_qc[split_name] = _build_split_artifacts(
            scanned_xml_files=scanned_xml_files,
            images=images,
            annotations=annotations,
            skipped_missing_images=skipped_missing_images,
            skipped_images_without_mapped_annotations=skipped_images_without_mapped_annotations,
            skipped_annotations_invalid_bbox=skipped_annotations_invalid_bbox,
            resolved_duplicate_filenames=resolved_duplicate_filenames,
            skipped_ambiguous_duplicate_filenames=skipped_ambiguous_duplicate_filenames,
            target_class_counts=target_counts,
            source_class_counts=source_counts,
            unmapped_source_class_counts=unmapped_source_counts,
        )

    if not used_targets:
        raise ValueError("Obstacle-Dataset import resolved zero mapped BA-v1 targets.")

    categories = _resolve_category_order(used_targets, contract_order)
    category_id_by_name = {name: idx + 1 for idx, name in enumerate(categories)}
    annotation_entries = [
        {
            "id": item["id"],
            "image_id": item["image_id"],
            "category_id": category_id_by_name[item["category_name"]],
            "bbox": item["bbox"],
            "area": item["area"],
            "iscrowd": item["iscrowd"],
        }
        for item in raw_annotation_entries
    ]
    category_entries = [
        {"id": category_id_by_name[name], "name": name}
        for name in categories
    ]

    combined_coco = {
        "images": image_entries,
        "annotations": annotation_entries,
        "categories": category_entries,
    }
    validate_coco(combined_coco, images_dir=out_images_root)

    combined_coco_path = write_coco(out_dir / "instances_ba_v1.coco.json", combined_coco)
    splits_path = write_splits(out_dir, split_image_ids)

    train_coco = build_split_coco(combined_coco, split_image_ids["train"])
    val_coco = build_split_coco(combined_coco, split_image_ids["val"])
    test_coco = build_split_coco(combined_coco, split_image_ids["test"])
    train_coco_path = write_coco(out_dir / "annotations_train.coco.json", train_coco)
    val_coco_path = write_coco(out_dir / "annotations_val.coco.json", val_coco)
    test_coco_path = write_coco(out_dir / "annotations_test.coco.json", test_coco)

    class_names_path = out_dir / "class_names.json"
    class_names_path.write_text(
        json.dumps(
            {
                "class_names": categories,
                "label_map": str(label_map_path),
                "source_format": "pascal_voc_split_xml",
                "dataset_dir": str(dataset_root),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    qc_report = {
        "source_dataset_dir": str(dataset_root),
        "source_format": "pascal_voc_split_xml",
        "image_resolution_source": "local_image_files",
        "image_resolution_roots": [
            str(path)
            for path in [
                dataset_root / "img-train",
                dataset_root / "img-val",
                dataset_root / "img-test",
                dataset_root / "JPEGImages",
                dataset_root / "OD-test" / "JPEGImages",
            ]
            if path.exists()
        ],
        "label_map": str(label_map_path),
        "categories": categories,
        "summary": {
            "images": len(image_entries),
            "annotations": len(annotation_entries),
            "categories": len(categories),
            "copied_images": copied,
            "symlinked_images": symlinked,
        },
        "splits": {
            split_name: {
                "scanned_xml_files": split_qc[split_name].scanned_xml_files,
                "exported_images": split_qc[split_name].images,
                "exported_annotations": split_qc[split_name].annotations,
                "skipped_missing_images": split_qc[split_name].skipped_missing_images,
                "skipped_images_without_mapped_annotations": split_qc[
                    split_name
                ].skipped_images_without_mapped_annotations,
                "skipped_annotations_invalid_bbox": split_qc[
                    split_name
                ].skipped_annotations_invalid_bbox,
                "resolved_duplicate_filenames": split_qc[split_name].resolved_duplicate_filenames,
                "skipped_ambiguous_duplicate_filenames": split_qc[
                    split_name
                ].skipped_ambiguous_duplicate_filenames,
                "target_class_counts": split_qc[split_name].target_class_counts,
                "source_class_counts": split_qc[split_name].source_class_counts,
                "unmapped_source_class_counts": split_qc[split_name].unmapped_source_class_counts,
                "missing_image_samples": missing_image_samples[split_name],
            }
            for split_name in ("train", "val", "test")
        },
    }
    qc_report_path = out_dir / "qc_report.json"
    qc_report_path.write_text(json.dumps(qc_report, indent=2), encoding="utf-8")

    return ObstacleDatasetImportArtifacts(
        out_dir=out_dir,
        images_dir=out_images_root,
        combined_coco_path=combined_coco_path,
        train_coco_path=train_coco_path,
        val_coco_path=val_coco_path,
        test_coco_path=test_coco_path,
        splits_path=splits_path,
        class_names_path=class_names_path,
        qc_report_path=qc_report_path,
        train=split_qc["train"],
        val=split_qc["val"],
        test=split_qc["test"],
        categories=categories,
    )
