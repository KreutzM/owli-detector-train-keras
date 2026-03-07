from pathlib import Path

from typer.testing import CliRunner

from owli_train.cli import app
from owli_train.golden.detect import (
    GoldenDetectArtifacts,
    MissingGoldenDependenciesError,
    build_golden_detect_config,
    build_golden_payload,
)
from owli_train.tflite_detect import TFLiteDetection

runner = CliRunner()


def test_golden_detect_cli_wires_flags(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MODELMAKER_PYTHON_EXE", raising=False)
    model = tmp_path / "model.tflite"
    model.write_bytes(b"fake")
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake")
    out = tmp_path / "golden.json"

    captured_cfg: dict[str, object] = {}

    def fake_build_golden_detect_config(**kwargs):
        captured_cfg.update(kwargs)
        return "CFG"

    def fake_generate(_cfg):
        return GoldenDetectArtifacts(out_path=out, num_detections=3)

    monkeypatch.setattr(
        "owli_train.cli.build_golden_detect_config", fake_build_golden_detect_config
    )
    monkeypatch.setattr("owli_train.cli.generate_golden_detect", fake_generate)

    result = runner.invoke(
        app,
        [
            "golden",
            "detect",
            "--model",
            str(model),
            "--image",
            str(image),
            "--out",
            str(out),
            "--score-threshold",
            "0.45",
            "--max-results",
            "9",
        ],
    )

    assert result.exit_code == 0
    assert "detections: 3" in result.stdout
    assert captured_cfg["model_path"] == model
    assert captured_cfg["image_path"] == image
    assert captured_cfg["out_path"] == out
    assert captured_cfg["score_threshold"] == 0.45
    assert captured_cfg["max_results"] == 9


def test_golden_detect_cli_delegates_with_required_flags(tmp_path: Path, monkeypatch):
    model = tmp_path / "model.tflite"
    model.write_bytes(b"fake")
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake")
    out = tmp_path / "golden.json"
    captured: dict[str, object] = {}

    def fake_delegate(args):
        captured["args"] = args
        return 0

    monkeypatch.setenv("MODELMAKER_PYTHON_EXE", "/tmp/fake-modelmaker-python")
    monkeypatch.setattr("owli_train.cli._delegate_to_modelmaker_python", fake_delegate)

    result = runner.invoke(
        app,
        [
            "golden",
            "detect",
            "--model",
            str(model),
            "--image",
            str(image),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    assert captured["args"] == [
        "golden",
        "detect",
        "--model",
        str(model),
        "--image",
        str(image),
        "--out",
        str(out),
        "--score-threshold",
        "0.3",
        "--max-results",
        "20",
    ]


def test_golden_detect_cli_dependency_message(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MODELMAKER_PYTHON_EXE", raising=False)
    model = tmp_path / "model.tflite"
    model.write_bytes(b"fake")
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake")
    out = tmp_path / "golden.json"

    def fake_build_golden_detect_config(**_kwargs):
        return "CFG"

    def fake_generate(_cfg):
        raise MissingGoldenDependenciesError(
            "Install with: pip install -r requirements\\modelmaker.txt"
        )

    monkeypatch.setattr(
        "owli_train.cli.build_golden_detect_config", fake_build_golden_detect_config
    )
    monkeypatch.setattr("owli_train.cli.generate_golden_detect", fake_generate)

    result = runner.invoke(
        app,
        [
            "golden",
            "detect",
            "--model",
            str(model),
            "--image",
            str(image),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 1
    assert "requirements\\modelmaker.txt" in result.stdout


def test_golden_payload_schema_with_stub_detections():
    cfg = build_golden_detect_config(
        model_path=Path("work/runs/example/artifacts/model.tflite"),
        image_path=Path("tests/smoke_coco/images/smoke1.jpg"),
        out_path=Path("work/golden/sample.json"),
        score_threshold=0.3,
        max_results=3,
    )
    payload = build_golden_payload(
        cfg=cfg,
        preprocess={
            "resize_policy": "letterbox_square",
            "target_size": 320,
            "input_shape": [1, 320, 320, 3],
            "input_dtype": "float32",
            "normalization": "[0,1]",
            "color_space": "RGB",
            "pad_value": 0,
        },
        detections=[
            TFLiteDetection(class_index=0, score=0.91, bbox_xywh=(1.0, 2.0, 30.0, 40.0)),
            TFLiteDetection(class_index=1, score=0.55, bbox_xywh=(3.0, 4.0, 10.0, 12.0)),
        ],
        class_names=["person", "car"],
        class_source="labels.txt",
        model_metadata={"input_size": 320, "bbox_format": "xywh"},
        inspect_summary={
            "builtin_ops_only": True,
            "inputs": [],
            "outputs": [],
            "operator_names": [],
        },
    )

    assert payload["contract"]["bbox_format"] == "xywh"
    assert payload["contract"]["coordinates"] == "absolute_pixels"
    assert payload["contract"]["class_labels_source"] == "labels.txt"
    assert payload["contract"]["input_preprocessing"]["resize_policy"] == "letterbox_square"
    assert len(payload["detections"]) == 2
    first = payload["detections"][0]
    assert first["class_name"] == "person"
    assert isinstance(first["score"], float)
    assert len(first["bbox"]) == 4
