from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def ensure_train_split_class_coverage(
    coco: dict[str, Any],
    splits: dict[str, list[int]],
) -> dict[str, list[int]]:
    split_lists = {name: list(splits[name]) for name in ("train", "val", "test")}
    split_by_image_id = {
        int(image_id): split_name for split_name, image_ids in split_lists.items() for image_id in image_ids
    }
    train_ids = {int(image_id) for image_id in split_lists["train"]}

    class_to_image_ids: dict[int, set[int]] = {}
    for annotation in coco["annotations"]:
        category_id = int(annotation["category_id"])
        image_id = int(annotation["image_id"])
        class_to_image_ids.setdefault(category_id, set()).add(image_id)

    missing_category_ids = {
        category_id
        for category_id, image_ids in class_to_image_ids.items()
        if image_ids and not (image_ids & train_ids)
    }
    if not missing_category_ids:
        return split_lists

    while missing_category_ids:
        candidate_image_ids = sorted(
            {
                image_id
                for category_id in missing_category_ids
                for image_id in class_to_image_ids[category_id]
                if image_id not in train_ids
            }
        )
        if not candidate_image_ids:
            break

        def _candidate_rank(image_id: int) -> tuple[int, int, int]:
            covers = sum(
                1 for category_id in missing_category_ids if image_id in class_to_image_ids[category_id]
            )
            source_split = split_by_image_id.get(image_id, "test")
            # Prefer pulling from test before val to keep validation less disturbed.
            split_priority = 1 if source_split == "test" else 0
            return (covers, split_priority, -image_id)

        selected_image_id = max(candidate_image_ids, key=_candidate_rank)
        source_split = split_by_image_id.get(selected_image_id)
        if source_split in {"val", "test"}:
            split_lists[source_split].remove(selected_image_id)
        elif source_split != "train":
            raise ValueError(f"Image id {selected_image_id} is not assigned to a known split.")
        split_lists["train"].append(selected_image_id)
        split_by_image_id[selected_image_id] = "train"
        train_ids.add(selected_image_id)
        missing_category_ids = {
            category_id
            for category_id, image_ids in class_to_image_ids.items()
            if image_ids and not (image_ids & train_ids)
        }

    return split_lists


def split_coco_image_ids(
    coco: dict[str, Any],
    seed: int = 1337,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    ensure_train_class_coverage: bool = False,
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
    splits = {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }
    if ensure_train_class_coverage:
        return ensure_train_split_class_coverage(coco, splits)
    return splits


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
