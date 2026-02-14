from pathlib import Path

from typer.testing import CliRunner

from owli_train.cli import app
from owli_train.training.keras_detector import (
    MissingTrainingDependenciesError,
    TrainingArtifacts,
)

runner = CliRunner()


def _write_min_config(path: Path) -> None:
    path.write_text(
        """
model:
  arch: yolov8
data:
  coco_json: data/coco/instances.json
  images_dir: data/coco/images
""",
        encoding="utf-8",
    )


def test_train_detect_cli_wires_runtime_flags(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "train.yaml"
    _write_min_config(cfg)

    captured: dict[str, object] = {}

    def fake_train_detector_from_config(**kwargs):
        captured.update(kwargs)
        return TrainingArtifacts(
            run_id="run-123",
            run_dir=Path("work/runs/run-123"),
            keras_model_path=Path("work/runs/run-123/artifacts/detector.keras"),
            saved_model_dir=Path("work/runs/run-123/artifacts/saved_model"),
            checkpoint_dir=Path("work/runs/run-123/checkpoints"),
            logs_dir=Path("work/runs/run-123/logs"),
        )

    monkeypatch.setattr(
        "owli_train.cli.train_detector_from_config", fake_train_detector_from_config
    )

    result = runner.invoke(
        app,
        [
            "train",
            "detect",
            "--config",
            str(cfg),
            "--run-name",
            "smoke",
            "--max-steps",
            "1",
            "--limit-train-images",
            "3",
            "--limit-val-images",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "run=run-123" in result.stdout
    assert captured["config_path"] == cfg
    assert captured["run_name"] == "smoke"
    assert captured["max_steps"] == 1
    assert captured["limit_train_images"] == 3
    assert captured["limit_val_images"] == 2


def test_train_detect_cli_dependency_error_message(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "train.yaml"
    _write_min_config(cfg)

    def raise_missing(**_kwargs):
        raise MissingTrainingDependenciesError(
            "Install with: pip install -r requirements\\keras.txt"
        )

    monkeypatch.setattr("owli_train.cli.train_detector_from_config", raise_missing)

    result = runner.invoke(app, ["train", "detect", "--config", str(cfg)])

    assert result.exit_code == 1
    assert "pip install -r requirements\\keras.txt" in result.stdout


def test_train_detect_cli_missing_coco_path_message(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    cfg = tmp_path / "train_missing_coco.yaml"
    cfg.write_text(
        f"""
model:
  arch: yolov8
data:
  coco_json: {tmp_path / "missing.json"}
  images_dir: {images_dir}
""",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["train", "detect", "--config", str(cfg), "--max-steps", "1"])

    assert result.exit_code == 1
    assert "data.coco_json was not found" in result.stdout
