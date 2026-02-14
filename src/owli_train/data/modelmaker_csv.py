from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from owli_train.data.coco import load_coco, validate_coco


@dataclass(frozen=True)
class ModelMakerCSVArtifacts:
    csv_path: Path
    class_names_path: Path
    rows: int
    images: int
    annotations: int


def _load_split_assignments(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("splits_json must be a JSON object.")

    assignments: dict[int, str] = {}
    for key, set_name in (("train", "TRAIN"), ("val", "VAL"), ("test", "TEST")):
        values = raw.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"splits_json key `{key}` must be a list.")
        for image_id in values:
            iid = int(image_id)
            if iid in assignments and assignments[iid] != set_name:
                raise ValueError(f"Image id {iid} appears in multiple split sets.")
            assignments[iid] = set_name
    return assignments


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def export_coco_to_modelmaker_csv(
    *,
    coco_path: Path,
    images_dir: Path,
    out_csv: Path,
    splits_json: Path | None = None,
    class_names_out: Path | None = None,
) -> ModelMakerCSVArtifacts:
    coco = load_coco(coco_path)
    summary = validate_coco(coco, images_dir=images_dir)

    images_by_id: dict[int, dict[str, Any]] = {int(image["id"]): image for image in coco["images"]}
    categories_by_id: dict[int, str] = {
        int(category["id"]): str(category["name"]) for category in coco["categories"]
    }
    split_assignments = _load_split_assignments(splits_json) if splits_json is not None else {}

    rows: list[list[str]] = []
    for annotation in sorted(coco["annotations"], key=lambda item: int(item["id"])):
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        image = images_by_id[image_id]
        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        if width <= 0 or height <= 0:
            image_path = images_dir / str(image["file_name"])
            from PIL import Image

            with Image.open(image_path) as img:
                width, height = img.size

        x, y, w, h = [float(v) for v in annotation["bbox"]]
        xmin = _clamp01(x / width)
        ymin = _clamp01(y / height)
        xmax = _clamp01((x + w) / width)
        ymax = _clamp01((y + h) / height)
        if xmax <= xmin or ymax <= ymin:
            continue

        rows.append(
            [
                split_assignments.get(image_id, "TRAIN"),
                str(image["file_name"]),
                categories_by_id[category_id],
                f"{xmin:.6f}",
                f"{ymin:.6f}",
                "",
                "",
                f"{xmax:.6f}",
                f"{ymax:.6f}",
            ]
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    class_names_path = (
        class_names_out if class_names_out is not None else out_csv.with_suffix(".class_names.json")
    )
    ordered_categories = sorted(coco["categories"], key=lambda cat: int(cat["id"]))
    class_names_path.write_text(
        json.dumps(
            {
                "class_names": [str(cat["name"]) for cat in ordered_categories],
                "category_ids": [int(cat["id"]) for cat in ordered_categories],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return ModelMakerCSVArtifacts(
        csv_path=out_csv,
        class_names_path=class_names_path,
        rows=len(rows),
        images=summary.images,
        annotations=summary.annotations,
    )
