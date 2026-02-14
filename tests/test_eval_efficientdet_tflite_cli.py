from pathlib import Path

from typer.testing import CliRunner

from owli_train.cli import app
from owli_train.eval.efficientdet_tflite import (
    EvalEfficientDetTFLiteArtifacts,
    MissingEfficientDetTFLiteEvalDependenciesError,
)

runner = CliRunner()


def test_eval_efficientdet_tflite_cli_wires_flags(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MODELMAKER_PYTHON_EXE", raising=False)
    coco = tmp_path / "eval.json"
    coco.write_text(
        '{"images": [], "annotations": [], "categories": [{"id": 1, "name": "person"}]}',
        encoding="utf-8",
    )
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    model = tmp_path / "model.tflite"
    model.write_bytes(b"fake")
    category_map = tmp_path / "category_map.yaml"
    category_map.write_text("0: person\n", encoding="utf-8")

    captured_cfg: dict[str, object] = {}

    def fake_build_eval_efficientdet_tflite_config(**kwargs):
        captured_cfg.update(kwargs)
        return "CFG"

    def fake_eval(cfg):
        assert cfg == "CFG"
        return EvalEfficientDetTFLiteArtifacts(
            json_report_path=tmp_path / "reports" / "eval_effdet.json",
            markdown_report_path=tmp_path / "reports" / "eval_effdet.md",
            model_path=model,
        )

    monkeypatch.setattr(
        "owli_train.cli.build_eval_efficientdet_tflite_config",
        fake_build_eval_efficientdet_tflite_config,
    )
    monkeypatch.setattr("owli_train.cli.evaluate_efficientdet_tflite", fake_eval)

    result = runner.invoke(
        app,
        [
            "eval",
            "efficientdet-tflite",
            "--coco",
            str(coco),
            "--images-dir",
            str(images_dir),
            "--model",
            str(model),
            "--limit-images",
            "7",
            "--score-threshold",
            "0.4",
            "--max-detections-per-image",
            "50",
            "--category-map",
            str(category_map),
        ],
    )

    assert result.exit_code == 0
    assert "report_json" in result.stdout
    assert captured_cfg["coco_path"] == coco
    assert captured_cfg["images_dir"] == images_dir
    assert captured_cfg["model_path"] == model
    assert captured_cfg["limit_images"] == 7
    assert captured_cfg["score_threshold"] == 0.4
    assert captured_cfg["max_detections_per_image"] == 50
    assert captured_cfg["category_map_path"] == category_map


def test_eval_efficientdet_tflite_cli_dependency_message(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MODELMAKER_PYTHON_EXE", raising=False)
    coco = tmp_path / "eval.json"
    coco.write_text(
        '{"images": [], "annotations": [], "categories": [{"id": 1, "name": "person"}]}',
        encoding="utf-8",
    )
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    model = tmp_path / "model.tflite"
    model.write_bytes(b"fake")

    def fake_build_eval_efficientdet_tflite_config(**_kwargs):
        return "CFG"

    def fake_eval(_cfg):
        raise MissingEfficientDetTFLiteEvalDependenciesError(
            "Install with: pip install -r requirements\\eval.txt and "
            "pip install -r requirements\\modelmaker.txt"
        )

    monkeypatch.setattr(
        "owli_train.cli.build_eval_efficientdet_tflite_config",
        fake_build_eval_efficientdet_tflite_config,
    )
    monkeypatch.setattr("owli_train.cli.evaluate_efficientdet_tflite", fake_eval)

    result = runner.invoke(
        app,
        [
            "eval",
            "efficientdet-tflite",
            "--coco",
            str(coco),
            "--images-dir",
            str(images_dir),
            "--model",
            str(model),
        ],
    )

    assert result.exit_code == 1
    assert "requirements\\eval.txt" in result.stdout
    assert "requirements\\modelmaker.txt" in result.stdout
