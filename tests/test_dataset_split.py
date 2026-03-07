import json
from pathlib import Path

from typer.testing import CliRunner

from owli_train.cli import app
from owli_train.data.split import ensure_train_split_class_coverage, split_coco_image_ids

runner = CliRunner()


def _make_coco(num_images: int = 10) -> dict:
    return {
        "images": [{"id": i, "file_name": f"{i}.jpg"} for i in range(1, num_images + 1)],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
            for i in range(1, num_images + 1)
        ],
        "categories": [{"id": 1, "name": "person"}],
    }


def test_split_deterministic_for_same_seed():
    coco = _make_coco()
    first = split_coco_image_ids(coco, seed=1337)
    second = split_coco_image_ids(coco, seed=1337)
    assert first == second


def test_ensure_train_split_class_coverage_moves_missing_class_into_train():
    coco = {
        "images": [{"id": idx, "file_name": f"{idx}.jpg"} for idx in range(1, 5)],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 4,
                "category_id": 2,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "bus"}],
    }
    splits = {"train": [1], "val": [2], "test": [3, 4]}

    repaired = ensure_train_split_class_coverage(coco, splits)

    assert repaired["train"] == [1, 4]
    assert repaired["val"] == [2]
    assert repaired["test"] == [3]


def test_split_can_optionally_ensure_train_class_coverage():
    coco = {
        "images": [{"id": idx, "file_name": f"{idx}.jpg"} for idx in range(1, 5)],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 3,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
            {
                "id": 4,
                "image_id": 4,
                "category_id": 2,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "bus"}],
    }

    plain = split_coco_image_ids(coco, seed=0, train_frac=0.25, val_frac=0.25)
    repaired = split_coco_image_ids(
        coco,
        seed=0,
        train_frac=0.25,
        val_frac=0.25,
        ensure_train_class_coverage=True,
    )

    assert plain["train"] == [3]
    assert repaired["train"] == [3, 4]
    assert 4 in repaired["train"]


def test_dataset_split_writes_split_json_and_coco_files(tmp_path: Path):
    coco_path = tmp_path / "instances.json"
    coco_path.write_text(json.dumps(_make_coco()), encoding="utf-8")

    out_dir = tmp_path / "splits"
    result = runner.invoke(
        app,
        [
            "dataset",
            "split",
            "--coco",
            str(coco_path),
            "--out-dir",
            str(out_dir),
            "--seed",
            "1337",
            "--write-coco",
        ],
    )

    assert result.exit_code == 0

    splits_path = out_dir / "splits.json"
    train_path = out_dir / "instances_train.json"
    val_path = out_dir / "instances_val.json"
    test_path = out_dir / "instances_test.json"

    assert splits_path.is_file()
    assert train_path.is_file()
    assert val_path.is_file()
    assert test_path.is_file()

    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    train = json.loads(train_path.read_text(encoding="utf-8"))
    val = json.loads(val_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))

    assert len(train["images"]) == len(splits["train"])
    assert len(val["images"]) == len(splits["val"])
    assert len(test["images"]) == len(splits["test"])


def test_dataset_split_cli_supports_train_class_coverage_flag(tmp_path: Path):
    coco_path = tmp_path / "instances.json"
    coco = {
        "images": [{"id": idx, "file_name": f"{idx}.jpg"} for idx in range(1, 5)],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 3,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
            {
                "id": 4,
                "image_id": 4,
                "category_id": 2,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "bus"}],
    }
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    out_dir = tmp_path / "splits"
    result = runner.invoke(
        app,
        [
            "dataset",
            "split",
            "--coco",
            str(coco_path),
            "--out-dir",
            str(out_dir),
            "--seed",
            "0",
            "--train-frac",
            "0.25",
            "--val-frac",
            "0.25",
            "--ensure-train-class-coverage",
        ],
    )

    assert result.exit_code == 0
    splits = json.loads((out_dir / "splits.json").read_text(encoding="utf-8"))
    assert splits["train"] == [3, 4]
