import importlib.util
from pathlib import Path

import pytest
from typer.testing import CliRunner

from owli_train.cli import app

runner = CliRunner()

HAS_TF = importlib.util.find_spec("tensorflow") is not None


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
