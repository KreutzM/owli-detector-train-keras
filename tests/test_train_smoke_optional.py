import importlib.util
from pathlib import Path

import pytest
from typer.testing import CliRunner

from owli_train.cli import app

runner = CliRunner()


HAS_TF = importlib.util.find_spec("tensorflow") is not None
HAS_KERAS_CV = importlib.util.find_spec("keras_cv") is not None


def _extract_run_dir(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("run_dir:"):
            return Path(line.split(":", 1)[1].strip())
    raise AssertionError("run_dir was not printed by train detect output.")


@pytest.mark.skipif(not (HAS_TF and HAS_KERAS_CV), reason="tensorflow/keras_cv not installed")
def test_train_detect_smoke_config_runs_with_tf():
    result = runner.invoke(
        app,
        [
            "train",
            "detect",
            "--config",
            "configs/train_detector_smoke.yaml",
            "--max-steps",
            "1",
            "--run-name",
            "pytest-smoke",
        ],
    )

    assert result.exit_code == 0
    assert "run=" in result.stdout
    assert "run_dir:" in result.stdout


@pytest.mark.skipif(not (HAS_TF and HAS_KERAS_CV), reason="tensorflow/keras_cv not installed")
def test_train_export_inspect_builtins_only_smoke_with_tf():
    train_result = runner.invoke(
        app,
        [
            "train",
            "detect",
            "--config",
            "configs/train_detector_builtins_smoke.yaml",
            "--max-steps",
            "1",
            "--run-name",
            "pytest-builtins-smoke",
        ],
    )
    assert train_result.exit_code == 0
    run_dir = _extract_run_dir(train_result.stdout)

    export_result = runner.invoke(
        app,
        [
            "export",
            "tflite",
            "--run-dir",
            str(run_dir),
            "--require-builtins-only",
            "--quant",
            "none",
        ],
    )
    assert export_result.exit_code == 0
    assert "builtin_ops_only: true" in export_result.stdout

    inspect_result = runner.invoke(
        app,
        [
            "inspect",
            "tflite",
            "--model",
            str(run_dir / "artifacts" / "detector.tflite"),
        ],
    )
    assert inspect_result.exit_code == 0
    assert "builtin_ops_only: true" in inspect_result.stdout
