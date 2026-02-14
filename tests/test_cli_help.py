from typer.testing import CliRunner

from owli_train.cli import app

runner = CliRunner()


def test_help():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0


def test_dataset_help_lists_normalize():
    r = runner.invoke(app, ["dataset", "--help"])
    assert r.exit_code == 0
    assert "normalize" in r.stdout


def test_train_detect_help_lists_smoke_flags():
    r = runner.invoke(app, ["train", "detect", "--help"])
    assert r.exit_code == 0
    assert "--max-steps" in r.stdout
    assert "--limit-train-images" in r.stdout
    assert "--limit-val-images" in r.stdout


def test_eval_detect_help_lists_eval_flags():
    r = runner.invoke(app, ["eval", "detect", "--help"])
    assert r.exit_code == 0
    assert "--run-dir" in r.stdout
    assert "--model" in r.stdout
    assert "--limit-images" in r.stdout
    assert "--score-threshold" in r.stdout


def test_export_tflite_help_lists_quant_and_rep_flags():
    r = runner.invoke(app, ["export", "tflite", "--help"])
    assert r.exit_code == 0
    assert "--run-dir" in r.stdout
    assert "--saved-model" in r.stdout
    assert "--model" in r.stdout
    assert "--quant" in r.stdout
    assert "--rep-coco" in r.stdout
    assert "--require-builtins-only" in r.stdout


def test_bench_tflite_help_lists_bench_flags():
    r = runner.invoke(app, ["bench", "tflite", "--help"])
    assert r.exit_code == 0
    assert "--run-dir" in r.stdout
    assert "--model" in r.stdout
    assert "--limit-images" in r.stdout
    assert "--warmup-runs" in r.stdout


def test_inspect_tflite_help_lists_flags():
    r = runner.invoke(app, ["inspect", "tflite", "--help"])
    assert r.exit_code == 0
    assert "--model" in r.stdout
