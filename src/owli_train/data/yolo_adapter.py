from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from owli_train.data.coco import validate_coco


@dataclass(frozen=True)
class YoloImportArtifacts:
    coco_path: Path
    class_names_path: Path
    images: int
    annotations: int
    categories: int


def _load_class_names(data_yaml: Path) -> list[str]:
    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("YOLO data yaml must be an object.")

    names = raw.get("names")
    if isinstance(names, list):
        values = [str(v).strip() for v in names]
        if not all(values):
            raise ValueError("YOLO names list must not contain empty entries.")
        return values

    if isinstance(names, dict):
        by_id: dict[int, str] = {}
        for key, value in names.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError("YOLO names mapping keys must be integer-like.") from exc
            class_name = str(value).strip()
            if not class_name:
                raise ValueError("YOLO names mapping values must be non-empty.")
            by_id[class_id] = class_name
        return [by_id[idx] for idx in sorted(by_id)]

    raise ValueError("YOLO data yaml must provide `names` as list or mapping.")


def _resolve_data_yaml(yolo_dir: Path, data_yaml: Path | None) -> Path:
    if data_yaml is not None:
        if not data_yaml.is_file():
            raise ValueError(f"--data-yaml was not found: {data_yaml}")
        return data_yaml

    preferred = yolo_dir / "coco128.yaml"
    if preferred.is_file():
        return preferred

    candidates = sorted(yolo_dir.glob("*.yaml")) + sorted(yolo_dir.glob("*.yml"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError("Could not find dataset yaml in YOLO directory (expected *.yaml).")
    raise ValueError("Multiple yaml files found. Pass --data-yaml explicitly.")


def _resolve_image_path(images_dir: Path, rel_label_path: Path) -> Path:
    stem = images_dir / rel_label_path.with_suffix("")
    allowed_suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for suffix in allowed_suffixes:
        for variant in (suffix, suffix.upper()):
            candidate = stem.with_suffix(variant)
            if candidate.is_file():
                return candidate

    parent = stem.parent
    if parent.is_dir():
        target_stem = stem.name.lower()
        allowed = set(allowed_suffixes)
        for candidate in parent.iterdir():
            if not candidate.is_file():
                continue
            if candidate.stem.lower() != target_stem:
                continue
            if candidate.suffix.lower() in allowed:
                return candidate

    raise ValueError(f"No image file found for label file: {rel_label_path}")


def _resolve_image_label_pairs(yolo_root: Path) -> tuple[list[tuple[Path, Path]], Path]:
    images_dir = yolo_root / "images"
    labels_dir = yolo_root / "labels"
    if images_dir.is_dir() and labels_dir.is_dir():
        return [(images_dir, labels_dir)], images_dir

    pairs: list[tuple[Path, Path]] = []
    for candidate_labels_dir in sorted(yolo_root.rglob("labels")):
        if not candidate_labels_dir.is_dir():
            continue
        candidate_images_dir = candidate_labels_dir.parent / "images"
        if candidate_images_dir.is_dir():
            pairs.append((candidate_images_dir, candidate_labels_dir))

    if pairs:
        return pairs, yolo_root

    raise ValueError(
        "Could not resolve YOLO images/labels layout. Expected either "
        "<root>/images + <root>/labels or split folders like <root>/train/images + <root>/train/labels."
    )


def _parse_label_line(
    line: str, label_path: Path, line_index: int
) -> tuple[int, float, float, float, float]:
    parts = line.split()
    if len(parts) < 5:
        raise ValueError(f"{label_path}:{line_index} must have at least 5 columns.")

    try:
        class_id = int(parts[0])
        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label_path}:{line_index} has invalid YOLO values.") from exc
    return class_id, cx, cy, w, h


def import_yolo_to_coco(
    *,
    yolo_dir: Path,
    out_path: Path,
    data_yaml: Path | None = None,
) -> YoloImportArtifacts:
    yolo_root = Path(yolo_dir)
    if not yolo_root.is_dir():
        raise ValueError(f"--yolo-dir was not found: {yolo_root}")

    image_label_pairs, images_root = _resolve_image_label_pairs(yolo_root)

    data_yaml_path = _resolve_data_yaml(yolo_root, data_yaml)
    class_names = _load_class_names(data_yaml_path)
    if not class_names:
        raise ValueError("Resolved zero class names from YOLO yaml.")

    categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(class_names)]
    image_entries: list[dict[str, Any]] = []
    annotation_entries: list[dict[str, Any]] = []

    annotation_id = 1
    image_id = 1
    label_sources: list[tuple[Path, Path, Path]] = []
    for images_dir, labels_dir in image_label_pairs:
        pair_label_files = sorted(path for path in labels_dir.rglob("*.txt") if path.is_file())
        for label_file in pair_label_files:
            label_sources.append((images_dir, labels_dir, label_file))

    if not label_sources:
        raise ValueError("No YOLO label files found in resolved labels directories.")

    seen_file_names: set[str] = set()
    for images_dir, labels_dir, label_file in label_sources:
        rel_label_path = label_file.relative_to(labels_dir)
        image_path = _resolve_image_path(images_dir, rel_label_path)
        rel_image_path = image_path.relative_to(images_root)
        file_name = rel_image_path.as_posix()
        if file_name in seen_file_names:
            raise ValueError(f"Duplicate image file_name resolved from labels: {file_name}")
        seen_file_names.add(file_name)

        with Image.open(image_path) as img:
            width, height = img.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Image has invalid dimensions: {image_path}")

        image_entries.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        raw_lines = label_file.read_text(encoding="utf-8").splitlines()
        for line_index, raw in enumerate(raw_lines, start=1):
            line = raw.strip()
            if not line:
                continue
            class_id, cx, cy, bw, bh = _parse_label_line(line, label_file, line_index)
            if class_id < 0 or class_id >= len(class_names):
                raise ValueError(f"{label_file}:{line_index} class id out of range: {class_id}")
            if bw <= 0.0 or bh <= 0.0:
                continue

            x_min = (cx - (bw / 2.0)) * width
            y_min = (cy - (bh / 2.0)) * height
            x_max = (cx + (bw / 2.0)) * width
            y_max = (cy + (bh / 2.0)) * height

            x_min = max(0.0, min(float(width), x_min))
            y_min = max(0.0, min(float(height), y_min))
            x_max = max(0.0, min(float(width), x_max))
            y_max = max(0.0, min(float(height), y_max))

            out_w = x_max - x_min
            out_h = y_max - y_min
            if out_w <= 0.0 or out_h <= 0.0:
                continue

            annotation_entries.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [x_min, y_min, out_w, out_h],
                    "area": out_w * out_h,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        image_id += 1

    coco = {
        "images": image_entries,
        "annotations": annotation_entries,
        "categories": categories,
    }
    validate_coco(coco, images_dir=images_root)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    class_names_path = out_path.parent / "class_names.json"
    class_names_path.write_text(
        json.dumps(
            {
                "class_names": class_names,
                "source_data_yaml": str(data_yaml_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return YoloImportArtifacts(
        coco_path=out_path,
        class_names_path=class_names_path,
        images=len(image_entries),
        annotations=len(annotation_entries),
        categories=len(categories),
    )
