from pathlib import Path

from typer.testing import CliRunner

from owli_train.cli import app
from owli_train.export.tflite_export import (
    InspectTFLiteArtifacts,
    MissingTFLiteDependenciesError,
)

runner = CliRunner()


def test_inspect_tflite_cli_wires_flags(tmp_path: Path, monkeypatch):
    model = tmp_path / "detector.tflite"
    model.write_bytes(b"fake")
    captured: dict[str, object] = {}

    def fake_build_inspect_tflite_config(**kwargs):
        captured.update(kwargs)
        return "CFG"

    def fake_inspect(_cfg):
        return InspectTFLiteArtifacts(
            model_path=model,
            builtin_ops_only=False,
            operator_names=["FlexMatrixInverse", "ADD"],
            inputs=[
                {"name": "serving_default_inputs:0", "shape": [1, 640, 640, 3], "dtype": "float32"}
            ],
            outputs=[{"name": "Identity", "shape": [1, 10, 6], "dtype": "float32"}],
        )

    monkeypatch.setattr(
        "owli_train.cli.build_inspect_tflite_config", fake_build_inspect_tflite_config
    )
    monkeypatch.setattr("owli_train.cli.run_inspect_tflite", fake_inspect)

    result = runner.invoke(app, ["inspect", "tflite", "--model", str(model)])

    assert result.exit_code == 0
    assert captured["model_path"] == model
    assert "builtin_ops_only: false" in result.stdout
    assert "FlexMatrixInverse" in result.stdout


def test_inspect_tflite_cli_dependency_message(tmp_path: Path, monkeypatch):
    model = tmp_path / "detector.tflite"
    model.write_bytes(b"fake")

    def fake_build_inspect_tflite_config(**_kwargs):
        return "CFG"

    def fake_inspect(_cfg):
        raise MissingTFLiteDependenciesError("Install with: pip install -r requirements\\keras.txt")

    monkeypatch.setattr(
        "owli_train.cli.build_inspect_tflite_config", fake_build_inspect_tflite_config
    )
    monkeypatch.setattr("owli_train.cli.run_inspect_tflite", fake_inspect)

    result = runner.invoke(app, ["inspect", "tflite", "--model", str(model)])

    assert result.exit_code == 1
    assert "requirements\\keras.txt" in result.stdout
