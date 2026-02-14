from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def split_coco_image_ids(
    coco: dict[str, Any],
    seed: int = 1337,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> dict[str, list[int]]:
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("Invalid fractions. Need train>0, val>0, train+val<1.")

    # Sort first so splits are deterministic even if input image order changes.
    ids = sorted(int(img["id"]) for img in coco["images"] if "id" in img)
    rnd = random.Random(seed)
    rnd.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def write_splits(out_dir: str | Path, splits: dict[str, list[int]]) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    out = p / "splits.json"
    out.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return out


def build_split_coco(coco: dict[str, Any], image_ids: list[int]) -> dict[str, Any]:
    image_id_set = set(image_ids)
    images = [img for img in coco["images"] if int(img["id"]) in image_id_set]
    annotations = [ann for ann in coco["annotations"] if int(ann["image_id"]) in image_id_set]

    out: dict[str, Any] = {
        "images": images,
        "annotations": annotations,
        "categories": list(coco["categories"]),
    }

    for extra_key in ("info", "licenses"):
        if extra_key in coco:
            out[extra_key] = coco[extra_key]
    return out


def write_split_coco_files(
    out_dir: str | Path,
    coco: dict[str, Any],
    splits: dict[str, list[int]],
) -> dict[str, Path]:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        split_obj = build_split_coco(coco, splits[split_name])
        split_path = p / f"instances_{split_name}.json"
        split_path.write_text(json.dumps(split_obj, indent=2), encoding="utf-8")
        written[split_name] = split_path
    return written
