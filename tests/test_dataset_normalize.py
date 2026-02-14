import json
from pathlib import Path

from typer.testing import CliRunner

from owli_train.cli import app

runner = CliRunner()


def test_dataset_normalize_writes_output(tmp_path: Path):
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 2,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "human"}],
    }
    coco_path = tmp_path / "instances.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    label_map = tmp_path / "label_map.yaml"
    label_map.write_text("map:\n  human: person\n", encoding="utf-8")

    out_path = tmp_path / "normalized.json"
    result = runner.invoke(
        app,
        [
            "dataset",
            "normalize",
            "--coco",
            str(coco_path),
            "--label-map",
            str(label_map),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0
    normalized = json.loads(out_path.read_text(encoding="utf-8"))
    assert normalized["categories"] == [{"id": 1, "name": "person"}]
    assert normalized["annotations"][0]["category_id"] == 1


def test_dataset_normalize_with_images_dir(tmp_path: Path):
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    coco_path = tmp_path / "instances.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "a.jpg").write_bytes(b"ok")

    out_path = tmp_path / "normalized.json"
    result = runner.invoke(
        app,
        [
            "dataset",
            "normalize",
            "--coco",
            str(coco_path),
            "--images-dir",
            str(images_dir),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0
    assert out_path.is_file()
