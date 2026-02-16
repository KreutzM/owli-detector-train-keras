import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from owli_train.cli import app

runner = CliRunner()


def test_dataset_import_yolo_converts_to_coco(tmp_path: Path):
    yolo_dir = tmp_path / "coco128"
    images_dir = yolo_dir / "images" / "train2017"
    labels_dir = yolo_dir / "labels" / "train2017"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    Image.new("RGB", (100, 50), color=(255, 0, 0)).save(images_dir / "0001.jpg")
    (labels_dir / "0001.txt").write_text("0 0.5 0.5 0.4 0.6\n", encoding="utf-8")
    (yolo_dir / "coco128.yaml").write_text("names:\n  - person\n", encoding="utf-8")

    out_path = tmp_path / "work" / "datasets" / "coco128" / "instances.json"
    result = runner.invoke(
        app,
        [
            "dataset",
            "import",
            "yolo",
            "--yolo-dir",
            str(yolo_dir),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0
    assert out_path.is_file()
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert payload["images"][0]["file_name"] == "train2017/0001.jpg"
    ann = payload["annotations"][0]
    assert ann["category_id"] == 1
    assert ann["bbox"] == [30.0, 10.0, 40.0, 30.0]

    class_names_path = out_path.parent / "class_names.json"
    assert class_names_path.is_file()


def test_dataset_import_yolo_supports_split_train_valid_layout(tmp_path: Path):
    yolo_dir = tmp_path / "obstacle4"
    train_images = yolo_dir / "train" / "images"
    train_labels = yolo_dir / "train" / "labels"
    valid_images = yolo_dir / "valid" / "images"
    valid_labels = yolo_dir / "valid" / "labels"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)
    valid_images.mkdir(parents=True)
    valid_labels.mkdir(parents=True)

    Image.new("RGB", (120, 80), color=(0, 255, 0)).save(train_images / "train_0001.jpg")
    Image.new("RGB", (64, 64), color=(0, 0, 255)).save(valid_images / "valid_0001.JPG")
    (train_labels / "train_0001.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (valid_labels / "valid_0001.txt").write_text("1 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (yolo_dir / "data.yaml").write_text("names:\n  - bump\n  - hole\n", encoding="utf-8")

    out_path = tmp_path / "work" / "datasets" / "obstacle4" / "instances.json"
    result = runner.invoke(
        app,
        [
            "dataset",
            "import",
            "yolo",
            "--yolo-dir",
            str(yolo_dir),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    file_names = sorted(item["file_name"] for item in payload["images"])
    assert file_names == ["train/images/train_0001.jpg", "valid/images/valid_0001.JPG"]
    assert len(payload["annotations"]) == 2
