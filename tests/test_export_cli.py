from pathlib import Path

from typer.testing import CliRunner

from owli_train.cli import app
from owli_train.export.tflite_export import (
    ExportTFLiteArtifacts,
    MissingTFLiteDependenciesError,
)

runner = CliRunner()


def test_export_tflite_cli_wires_flags(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rep_images = tmp_path / "images"
    rep_images.mkdir()
    rep_coco = tmp_path / "rep.json"
    rep_coco.write_text(
        '{"images": [], "annotations": [], "categories": [{"id": 1, "name": "person"}]}',
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_build_export_tflite_config(**kwargs):
        captured.update(kwargs)
        return "CFG"

    def fake_export(_cfg):
        return ExportTFLiteArtifacts(
            model_path=run_dir / "artifacts" / "saved_model",
            tflite_path=run_dir / "artifacts" / "detector.tflite",
            metadata_path=run_dir / "artifacts" / "detector.tflite.meta.json",
            quant="fp16",
            source_type="saved_model",
            builtin_ops_only=True,
        )

    monkeypatch.setattr(
        "owli_train.cli.build_export_tflite_config", fake_build_export_tflite_config
    )
    monkeypatch.setattr("owli_train.cli.run_export_tflite", fake_export)

    result = runner.invoke(
        app,
        [
            "export",
            "tflite",
            "--run-dir",
            str(run_dir),
            "--quant",
            "fp16",
            "--rep-coco",
            str(rep_coco),
            "--rep-images-dir",
            str(rep_images),
            "--rep-max-images",
            "4",
            "--require-builtins-only",
        ],
    )

    assert result.exit_code == 0
    normalized_stdout = result.stdout.replace("\n", "")
    assert "detector.tflite" in normalized_stdout
    assert captured["run_dir"] == run_dir
    assert captured["quant"] == "fp16"
    assert captured["rep_coco"] == rep_coco
    assert captured["rep_images_dir"] == rep_images
    assert captured["rep_max_images"] == 4
    assert captured["require_builtins_only"] is True


def test_export_tflite_cli_dependency_message(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_build_export_tflite_config(**_kwargs):
        return "CFG"

    def fake_export(_cfg):
        raise MissingTFLiteDependenciesError("Install with: pip install -r requirements\\keras.txt")

    monkeypatch.setattr(
        "owli_train.cli.build_export_tflite_config", fake_build_export_tflite_config
    )
    monkeypatch.setattr("owli_train.cli.run_export_tflite", fake_export)

    result = runner.invoke(app, ["export", "tflite", "--run-dir", str(run_dir)])

    assert result.exit_code == 1
    assert "requirements\\keras.txt" in result.stdout
