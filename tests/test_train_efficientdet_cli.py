import json
from pathlib import Path

from typer.testing import CliRunner

from owli_train.cli import app
from owli_train.training.modelmaker_efficientdet import (
    EfficientDetArtifacts,
    MissingModelMakerDependenciesError,
)

runner = CliRunner()


def _write_min_config(path: Path) -> None:
    path.write_text(
        """
model:
  variant: lite2
data:
  csv: work/datasets/coco128/modelmaker.csv
  images_dir: data/coco128/images/train2017
train:
  epochs: 1
  batch_size: 1
""",
        encoding="utf-8",
    )


def test_train_efficientdet_cli_wires_flags(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "effdet.yaml"
    _write_min_config(cfg)
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "label_alignment": {
                    "missing_classes_from_train_split": ["bus"],
                    "missing_classes_from_training": ["bus", "truck"],
                }
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_train_efficientdet_from_config(**kwargs):
        captured.update(kwargs)
        return EfficientDetArtifacts(
            run_id="run-123",
            run_dir=Path("work/runs/run-123"),
            tflite_path=Path("work/runs/run-123/artifacts/model.tflite"),
            labels_path=Path("work/runs/run-123/artifacts/labels.txt"),
            config_snapshot_path=Path("work/runs/run-123/config.yaml"),
            mapping_snapshot_path=mapping_path,
        )

    monkeypatch.setattr(
        "owli_train.cli.train_efficientdet_from_config", fake_train_efficientdet_from_config
    )

    result = runner.invoke(
        app,
        [
            "train",
            "efficientdet",
            "--config",
            str(cfg),
            "--variant",
            "lite3",
            "--run-name",
            "smoke",
            "--max-steps",
            "1",
            "--subset-seed",
            "2026",
            "--require-gpu",
        ],
    )

    assert result.exit_code == 0
    assert "run=run-123" in result.stdout
    assert "expected classes missing from TRAIN split: bus" in result.stdout
    assert "classes missing from training rows: bus, truck" in result.stdout
    assert captured["config_path"] == cfg
    assert captured["variant"] == "lite3"
    assert captured["run_name"] == "smoke"
    assert captured["max_steps"] == 1
    assert captured["subset_seed"] == 2026
    assert captured["require_gpu"] is True


def test_train_efficientdet_cli_dependency_message(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "effdet.yaml"
    _write_min_config(cfg)

    def raise_missing(**_kwargs):
        raise MissingModelMakerDependenciesError(
            "Install with: pip install -r requirements\\modelmaker.txt"
        )

    monkeypatch.setattr("owli_train.cli.train_efficientdet_from_config", raise_missing)
    result = runner.invoke(app, ["train", "efficientdet", "--config", str(cfg)])

    assert result.exit_code == 1
    assert "requirements\\modelmaker.txt" in result.stdout
