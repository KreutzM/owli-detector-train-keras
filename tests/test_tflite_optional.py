import importlib.util
from pathlib import Path

import pytest
from typer.testing import CliRunner

from owli_train.cli import app

runner = CliRunner()

HAS_TF = importlib.util.find_spec("tensorflow") is not None


def _build_flex_tflite_bytes(tf) -> bytes:
    inputs = tf.keras.Input(shape=(2, 2), dtype=tf.float32)
    outputs = tf.keras.layers.Lambda(lambda x: tf.linalg.inv(x), output_shape=(2, 2))(inputs)
    model = tf.keras.Model(inputs, outputs)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    return converter.convert()


@pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")
def test_export_and_bench_tflite_optional_smoke(tmp_path: Path):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(4, 3, activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2),
        ]
    )

    keras_path = tmp_path / "tiny.keras"
    model.save(keras_path)

    tflite_path = tmp_path / "tiny.tflite"
    export_result = runner.invoke(
        app,
        [
            "export",
            "tflite",
            "--model",
            str(keras_path),
            "--out",
            str(tflite_path),
            "--quant",
            "none",
        ],
    )

    assert export_result.exit_code == 0
    assert tflite_path.is_file()
    assert (tmp_path / "tiny.tflite.meta.json").is_file()

    bench_out = tmp_path / "bench.json"
    bench_result = runner.invoke(
        app,
        [
            "bench",
            "tflite",
            "--model",
            str(tflite_path),
            "--limit-images",
            "2",
            "--warmup-runs",
            "1",
            "--runs",
            "2",
            "--out",
            str(bench_out),
        ],
    )

    assert bench_result.exit_code == 0
    assert bench_out.is_file()


@pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")
def test_export_require_builtins_only_fails_for_flex_model(tmp_path: Path, monkeypatch):
    import tensorflow as tf

    flex_bytes = _build_flex_tflite_bytes(tf)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(4, 3, activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2),
        ]
    )
    keras_path = tmp_path / "tiny.keras"
    model.save(keras_path)

    def fake_convert_from_keras_model(*, tf, keras_path, cfg, input_size):
        return flex_bytes, 64

    monkeypatch.setattr(
        "owli_train.export.tflite_export._convert_from_keras_model", fake_convert_from_keras_model
    )

    result = runner.invoke(
        app,
        [
            "export",
            "tflite",
            "--model",
            str(keras_path),
            "--out",
            str(tmp_path / "blocked.tflite"),
            "--require-builtins-only",
        ],
    )

    assert result.exit_code == 1
    assert "SELECT_TF_OPS" in result.stdout
    assert "--require-builtins-only" in result.stdout


@pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")
def test_inspect_tflite_optional_reports_flex_ops(tmp_path: Path):
    import tensorflow as tf

    flex_path = tmp_path / "flex_model.tflite"
    flex_path.write_bytes(_build_flex_tflite_bytes(tf))

    result = runner.invoke(app, ["inspect", "tflite", "--model", str(flex_path)])

    assert result.exit_code == 0
    assert "builtin_ops_only: false" in result.stdout
    assert "FlexMatrixInverse" in result.stdout
    assert "inputs:" in result.stdout
    assert "outputs:" in result.stdout
