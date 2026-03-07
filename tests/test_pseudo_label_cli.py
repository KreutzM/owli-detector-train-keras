from pathlib import Path

from typer.testing import CliRunner

from owli_train.cli import app
from owli_train.pseudo_label.teacher_tfhub import (
    MissingTeacherDependenciesError,
    TeacherPseudoLabelArtifacts,
    TeacherPseudoLabelConfigError,
    build_teacher_pseudo_label_config,
)

runner = CliRunner()


def test_dataset_pseudo_label_coco_help_lists_flags() -> None:
    result = runner.invoke(app, ["dataset", "pseudo-label", "coco", "--help"])
    assert result.exit_code == 0
    assert "--images-dir" in result.stdout
    assert "--out" in result.stdout
    assert "--teacher" in result.stdout
    assert "--teacher-savedmodel" in result.stdout
    assert "--batch-size" in result.stdout
    assert "--score-threshold" in result.stdout


def test_dataset_pseudo_label_coco_delegates_with_required_flags(
    tmp_path: Path, monkeypatch
) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    out = tmp_path / "pseudo.json"
    captured: dict[str, object] = {}

    def fake_delegate(args):
        captured["args"] = args
        return 0

    monkeypatch.setenv("TEACHER_PYTHON_EXE", "/tmp/fake-teacher-python")
    monkeypatch.setattr("owli_train.cli._delegate_to_teacher_python", fake_delegate)

    result = runner.invoke(
        app,
        [
            "dataset",
            "pseudo-label",
            "coco",
            "--images-dir",
            str(images_dir),
            "--out",
            str(out),
            "--batch-size",
            "4",
        ],
    )

    assert result.exit_code == 0
    assert captured["args"] == [
        "dataset",
        "pseudo-label",
        "coco",
        "--images-dir",
        str(images_dir),
        "--out",
        str(out),
        "--teacher",
        "https://tfhub.dev/tensorflow/efficientdet/d2/1",
        "--batch-size",
        "4",
        "--input-size",
        "640",
        "--score-threshold",
        "0.6",
        "--max-detections-per-image",
        "50",
        "--seed",
        "1337",
        "--num-parallel-calls",
        "4",
        "--prefetch-buffer",
        "2",
        "--viz-max-images",
        "25",
    ]


def test_dataset_pseudo_label_coco_missing_images_dir_is_friendly(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("TEACHER_PYTHON_EXE", raising=False)
    out = tmp_path / "pseudo.json"
    missing_dir = tmp_path / "missing_images"
    result = runner.invoke(
        app,
        [
            "dataset",
            "pseudo-label",
            "coco",
            "--images-dir",
            str(missing_dir),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 1
    assert "--images-dir is not a directory" in result.stdout


def test_dataset_pseudo_label_coco_wires_config_and_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("TEACHER_PYTHON_EXE", raising=False)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "a.jpg").write_bytes(b"fake")
    out = tmp_path / "pseudo.json"
    report = tmp_path / "pseudo_report.json"

    captured_cfg: dict[str, object] = {}

    def fake_build_teacher_pseudo_label_config(**kwargs):
        captured_cfg.update(kwargs)
        return "CFG"

    def fake_generate_teacher_pseudo_labels(cfg):
        assert cfg == "CFG"
        return TeacherPseudoLabelArtifacts(
            pseudo_coco_path=out,
            report_path=report,
            images_processed=1,
            detections_kept=2,
            teacher_source="tfhub://dummy",
            elapsed_seconds=0.25,
        )

    monkeypatch.setattr(
        "owli_train.cli.build_teacher_pseudo_label_config",
        fake_build_teacher_pseudo_label_config,
    )
    monkeypatch.setattr(
        "owli_train.cli.generate_teacher_pseudo_labels",
        fake_generate_teacher_pseudo_labels,
    )

    result = runner.invoke(
        app,
        [
            "dataset",
            "pseudo-label",
            "coco",
            "--images-dir",
            str(images_dir),
            "--out",
            str(out),
            "--batch-size",
            "8",
            "--input-size",
            "640",
            "--score-threshold",
            "0.7",
            "--max-detections-per-image",
            "20",
            "--classes",
            "person,car",
            "--limit-images",
            "5",
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0
    assert "pseudo_coco:" in result.stdout
    assert captured_cfg["images_dir"] == images_dir
    assert captured_cfg["out_path"] == out
    assert captured_cfg["batch_size"] == 8
    assert captured_cfg["input_size"] == 640
    assert captured_cfg["score_threshold"] == 0.7
    assert captured_cfg["max_detections_per_image"] == 20
    assert captured_cfg["classes_filter"] == "person,car"
    assert captured_cfg["limit_images"] == 5
    assert captured_cfg["seed"] == 42


def test_dataset_pseudo_label_coco_dependency_message(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TEACHER_PYTHON_EXE", raising=False)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    out = tmp_path / "pseudo.json"

    monkeypatch.setattr(
        "owli_train.cli.build_teacher_pseudo_label_config",
        lambda **_kwargs: "CFG",
    )

    def _raise_missing(_cfg):
        raise MissingTeacherDependenciesError(
            "Teacher pseudo-labeling requires TensorFlow + TF Hub. Install with: "
            "pip install -r requirements\\teacher.txt"
        )

    monkeypatch.setattr("owli_train.cli.generate_teacher_pseudo_labels", _raise_missing)

    result = runner.invoke(
        app,
        [
            "dataset",
            "pseudo-label",
            "coco",
            "--images-dir",
            str(images_dir),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 1
    assert "requirements\\teacher.txt" in result.stdout
    assert "TEACHER_PYTHON_EXE" in result.stdout


def test_build_teacher_pseudo_label_config_default_report_path(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    out = tmp_path / "pseudo.json"

    cfg = build_teacher_pseudo_label_config(
        images_dir=images_dir,
        out_path=out,
        teacher="https://tfhub.dev/tensorflow/efficientdet/d2/1",
        teacher_savedmodel=None,
        batch_size=8,
        input_size=640,
        score_threshold=0.6,
        max_detections_per_image=50,
        classes_filter=None,
        limit_images=None,
        seed=1337,
        num_parallel_calls=4,
        prefetch_buffer=2,
        debug_io=False,
        report_out_path=None,
        viz_out_dir=None,
        viz_max_images=25,
    )

    assert cfg.report_out_path == tmp_path / "pseudo.report.json"


def test_build_teacher_pseudo_label_config_requires_json_out(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    try:
        build_teacher_pseudo_label_config(
            images_dir=images_dir,
            out_path=tmp_path / "pseudo.txt",
            teacher="https://tfhub.dev/tensorflow/efficientdet/d2/1",
            teacher_savedmodel=None,
            batch_size=8,
            input_size=640,
            score_threshold=0.6,
            max_detections_per_image=50,
            classes_filter=None,
            limit_images=None,
            seed=1337,
            num_parallel_calls=4,
            prefetch_buffer=2,
            debug_io=False,
            report_out_path=None,
            viz_out_dir=None,
            viz_max_images=25,
        )
    except TeacherPseudoLabelConfigError as exc:
        assert "--out must be a .json path." in str(exc)
    else:
        raise AssertionError("Expected TeacherPseudoLabelConfigError")
