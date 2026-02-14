import json
from pathlib import Path

from typer.testing import CliRunner

from owli_train.cli import app
from owli_train.data.split import split_coco_image_ids

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
