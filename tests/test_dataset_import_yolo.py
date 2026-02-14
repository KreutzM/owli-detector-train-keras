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
