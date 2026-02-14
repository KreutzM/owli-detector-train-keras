import csv
import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from owli_train.cli import app

runner = CliRunner()


def test_dataset_export_modelmaker_csv_writes_rows(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (200, 100), color=(0, 0, 0)).save(images_dir / "a.jpg")

    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 200, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [20, 10, 40, 20],
                "area": 800,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    coco_path = tmp_path / "instances.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    csv_path = tmp_path / "dataset.csv"
    result = runner.invoke(
        app,
        [
            "dataset",
            "export",
            "modelmaker-csv",
            "--coco",
            str(coco_path),
            "--images-dir",
            str(images_dir),
            "--out",
            str(csv_path),
        ],
    )

    assert result.exit_code == 0
    assert csv_path.is_file()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row[0] == "TRAIN"
    assert row[1] == "a.jpg"
    assert row[2] == "person"
    assert row[3] == "0.100000"
    assert row[4] == "0.100000"
    assert row[7] == "0.300000"
    assert row[8] == "0.300000"

    assert csv_path.with_suffix(".class_names.json").is_file()
