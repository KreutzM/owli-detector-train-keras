from pathlib import Path

import pytest

from owli_train.training.modelmaker_efficientdet import load_efficientdet_config


def test_load_efficientdet_config_parses_schema(tmp_path: Path):
    cfg_path = tmp_path / "effdet.yaml"
    cfg_path.write_text(
        """
model:
  variant: lite2
data:
  csv: work/datasets/coco128/modelmaker.csv
  images_dir: data/coco128/images/train2017
  label_map_json: work/datasets/coco128/class_names.json
train:
  epochs: 3
  batch_size: 2
  train_whole_model: true
outputs:
  work_dir: work
  out_dir: outputs
""",
        encoding="utf-8",
    )

    cfg = load_efficientdet_config(cfg_path)
    assert cfg.model.variant == "lite2"
    assert cfg.data.csv_path == Path("work/datasets/coco128/modelmaker.csv")
    assert cfg.train.epochs == 3
    assert cfg.train.batch_size == 2
    assert cfg.train.train_whole_model is True


def test_load_efficientdet_config_validates_batch_size(tmp_path: Path):
    cfg_path = tmp_path / "effdet_invalid.yaml"
    cfg_path.write_text(
        """
data:
  csv: a.csv
  images_dir: images
train:
  batch_size: 0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="batch_size"):
        load_efficientdet_config(cfg_path)
