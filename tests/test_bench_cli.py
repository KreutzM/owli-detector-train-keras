from pathlib import Path

from typer.testing import CliRunner

from owli_train.cli import app
from owli_train.export.tflite_export import (
    BenchTFLiteArtifacts,
    MissingTFLiteDependenciesError,
)

runner = CliRunner()


def test_bench_tflite_cli_wires_flags(tmp_path: Path, monkeypatch):
    model = tmp_path / "detector.tflite"
    model.write_bytes(b"fake")

    captured: dict[str, object] = {}

    def fake_build_bench_tflite_config(**kwargs):
        captured.update(kwargs)
        return "CFG"

    def fake_bench(_cfg):
        return BenchTFLiteArtifacts(
            model_path=model,
            report_path=tmp_path / "reports" / "bench_tflite.json",
        )

    monkeypatch.setattr("owli_train.cli.build_bench_tflite_config", fake_build_bench_tflite_config)
    monkeypatch.setattr("owli_train.cli.run_bench_tflite", fake_bench)

    result = runner.invoke(
        app,
        [
            "bench",
            "tflite",
            "--model",
            str(model),
            "--limit-images",
            "2",
            "--warmup-runs",
            "1",
            "--runs",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert "bench_tflite.json" in result.stdout
    assert captured["model_path"] == model
    assert captured["limit_images"] == 2
    assert captured["warmup_runs"] == 1
    assert captured["runs"] == 3


def test_bench_tflite_cli_dependency_message(tmp_path: Path, monkeypatch):
    model = tmp_path / "detector.tflite"
    model.write_bytes(b"fake")

    def fake_build_bench_tflite_config(**_kwargs):
        return "CFG"

    def fake_bench(_cfg):
        raise MissingTFLiteDependenciesError("Install with: pip install -r requirements\\keras.txt")

    monkeypatch.setattr("owli_train.cli.build_bench_tflite_config", fake_build_bench_tflite_config)
    monkeypatch.setattr("owli_train.cli.run_bench_tflite", fake_bench)

    result = runner.invoke(app, ["bench", "tflite", "--model", str(model)])

    assert result.exit_code == 1
    assert "requirements\\keras.txt" in result.stdout
