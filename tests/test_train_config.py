from pathlib import Path

import pytest

from owli_train.training.keras_detector import load_train_detect_config


def test_load_train_detect_config_parses_new_schema(tmp_path: Path):
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        """
model:
  arch: yolov8
  backbone: yolo_v8_xs_backbone
  input_size: 320
  class_names: [person, car]
  bounding_box_format: xywh
data:
  coco_json: data/coco/instances.json
  images_dir: data/coco/images
train:
  seed: 123
  batch_size: 4
  epochs: 2
  lr: 0.001
  callbacks:
    checkpoint: true
    tensorboard: false
outputs:
  work_dir: work
  out_dir: outputs
""",
        encoding="utf-8",
    )

    cfg = load_train_detect_config(cfg_path)

    assert cfg.model.arch == "yolov8"
    assert cfg.model.num_classes == 2
    assert cfg.model.class_names == ["person", "car"]
    assert cfg.data.coco_json == Path("data/coco/instances.json")
    assert cfg.data.images_dir == Path("data/coco/images")
    assert cfg.train.seed == 123
    assert cfg.train.batch_size == 4
    assert cfg.train.callbacks.tensorboard is False


def test_load_train_detect_config_requires_data_source(tmp_path: Path):
    cfg_path = tmp_path / "invalid.yaml"
    cfg_path.write_text(
        """
model:
  arch: yolov8
data:
  images_dir: data/coco/images
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Provide either data.coco_json"):
        load_train_detect_config(cfg_path)


def test_load_train_detect_config_rejects_class_count_mismatch(tmp_path: Path):
    cfg_path = tmp_path / "invalid_classes.yaml"
    cfg_path.write_text(
        """
model:
  arch: yolov8
  num_classes: 3
  class_names: [person, car]
data:
  coco_json: data/coco/instances.json
  images_dir: data/coco/images
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="num_classes"):
        load_train_detect_config(cfg_path)
