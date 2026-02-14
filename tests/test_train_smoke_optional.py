import importlib.util

import pytest
from typer.testing import CliRunner

from owli_train.cli import app

runner = CliRunner()


HAS_TF = importlib.util.find_spec("tensorflow") is not None
HAS_KERAS_CV = importlib.util.find_spec("keras_cv") is not None


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
