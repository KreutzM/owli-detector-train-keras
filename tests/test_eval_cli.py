from pathlib import Path

from typer.testing import CliRunner

from owli_train.cli import app
from owli_train.eval.detect import EvalArtifacts, MissingEvalDependenciesError

runner = CliRunner()


def test_eval_detect_cli_wires_flags(tmp_path: Path, monkeypatch):
    coco = tmp_path / "eval.json"
    coco.write_text(
        '{"images": [], "annotations": [], "categories": [{"id": 1, "name": "person"}]}',
        encoding="utf-8",
    )
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    category_map = tmp_path / "category_map.json"
    category_map.write_text('{"0": 1}', encoding="utf-8")

    captured_cfg = {}

    def fake_build_eval_detect_config(**kwargs):
        captured_cfg.update(kwargs)
        return "CFG"

    def fake_evaluate_detect(cfg):
        assert cfg == "CFG"
        return EvalArtifacts(
            json_report_path=tmp_path / "report" / "eval.json",
            markdown_report_path=tmp_path / "report" / "eval.md",
            model_path=run_dir / "artifacts" / "detector.keras",
            run_dir=run_dir,
        )

    monkeypatch.setattr("owli_train.cli.build_eval_detect_config", fake_build_eval_detect_config)
    monkeypatch.setattr("owli_train.cli.evaluate_detect", fake_evaluate_detect)

    result = runner.invoke(
        app,
        [
            "eval",
            "detect",
            "--coco",
            str(coco),
            "--images-dir",
            str(images_dir),
            "--run-dir",
            str(run_dir),
            "--limit-images",
            "5",
            "--score-threshold",
            "0.3",
            "--max-detections-per-image",
            "33",
            "--category-map",
            str(category_map),
        ],
    )

    assert result.exit_code == 0
    assert "report_json" in result.stdout
    assert captured_cfg["coco_path"] == coco
    assert captured_cfg["images_dir"] == images_dir
    assert captured_cfg["run_dir"] == run_dir
    assert captured_cfg["limit_images"] == 5
    assert captured_cfg["score_threshold"] == 0.3
    assert captured_cfg["max_detections_per_image"] == 33
    assert captured_cfg["category_map_path"] == category_map


def test_eval_detect_cli_dependency_message(tmp_path: Path, monkeypatch):
    coco = tmp_path / "eval.json"
    coco.write_text(
        '{"images": [], "annotations": [], "categories": [{"id": 1, "name": "person"}]}',
        encoding="utf-8",
    )
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_build_eval_detect_config(**_kwargs):
        return "CFG"

    def fake_evaluate_detect(_cfg):
        raise MissingEvalDependenciesError(
            "Install with: pip install -r requirements\\keras.txt and pip install -r requirements\\eval.txt"
        )

    monkeypatch.setattr("owli_train.cli.build_eval_detect_config", fake_build_eval_detect_config)
    monkeypatch.setattr("owli_train.cli.evaluate_detect", fake_evaluate_detect)

    result = runner.invoke(
        app,
        [
            "eval",
            "detect",
            "--coco",
            str(coco),
            "--images-dir",
            str(images_dir),
            "--run-dir",
            str(run_dir),
        ],
    )

    assert result.exit_code == 1
    assert "requirements\\eval.txt" in result.stdout
