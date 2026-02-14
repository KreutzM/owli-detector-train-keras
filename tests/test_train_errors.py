from pathlib import Path

import pytest

from owli_train.training.keras_detector import TrainingError, train_detector_from_config


def test_missing_coco_json_has_friendly_error(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    cfg = tmp_path / "train_missing_coco.yaml"
    cfg.write_text(
        f"""
model:
  arch: yolov8
data:
  coco_json: {tmp_path / "does_not_exist.json"}
  images_dir: {images_dir}
""",
        encoding="utf-8",
    )

    with pytest.raises(TrainingError, match="data.coco_json was not found"):
        train_detector_from_config(cfg, max_steps=1)


def test_missing_images_dir_has_friendly_error(tmp_path: Path):
    coco_path = tmp_path / "instances.json"
    coco_path.write_text(
        """
{
  "images": [{"id": 1, "file_name": "a.jpg"}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}],
  "categories": [{"id": 1, "name": "person"}]
}
""",
        encoding="utf-8",
    )

    cfg = tmp_path / "train_missing_images.yaml"
    cfg.write_text(
        f"""
model:
  arch: yolov8
data:
  coco_json: {coco_path}
  images_dir: {tmp_path / "missing_images"}
""",
        encoding="utf-8",
    )

    with pytest.raises(TrainingError, match="data.images_dir was not found"):
        train_detector_from_config(cfg, max_steps=1)
