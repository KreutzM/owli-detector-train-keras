from typer.testing import CliRunner

from owli_train.cli import app

runner = CliRunner()


def test_help():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "golden" in r.stdout


def test_dataset_help_lists_normalize():
    r = runner.invoke(app, ["dataset", "--help"])
    assert r.exit_code == 0
    assert "normalize" in r.stdout
    assert "balance-coco" in r.stdout
    assert "import" in r.stdout
    assert "export" in r.stdout
    assert "merge" in r.stdout
    assert "materialize-images" in r.stdout
    assert "pseudo-label" in r.stdout


def test_dataset_import_yolo_help_lists_flags():
    r = runner.invoke(app, ["dataset", "import", "yolo", "--help"])
    assert r.exit_code == 0
    assert "--yolo-dir" in r.stdout
    assert "--data-yaml" in r.stdout


def test_dataset_import_mapillary_vistas_help_lists_flags():
    r = runner.invoke(app, ["dataset", "import", "mapillary-vistas", "--help"])
    assert r.exit_code == 0
    assert "--mapillary-dir" in r.stdout
    assert "--out-dir" in r.stdout
    assert "--label-map" in r.stdout
    assert "--max-long-side" in r.stdout
    assert "--annotation-version" in r.stdout
    assert "limit-images-per-spl" in r.stdout


def test_dataset_import_obstacle_dataset_help_lists_flags():
    r = runner.invoke(app, ["dataset", "import", "obstacle-dataset", "--help"])
    assert r.exit_code == 0
    assert "--dataset-dir" in r.stdout
    assert "--out-dir" in r.stdout
    assert "--label-map" in r.stdout
    assert "--mode" in r.stdout


def test_dataset_export_modelmaker_csv_help_lists_flags():
    r = runner.invoke(app, ["dataset", "export", "modelmaker-csv", "--help"])
    assert r.exit_code == 0
    assert "--coco" in r.stdout
    assert "--images-dir" in r.stdout
    assert "--out" in r.stdout


def test_dataset_balance_coco_help_lists_flags():
    r = runner.invoke(app, ["dataset", "balance-coco", "--help"])
    assert r.exit_code == 0
    assert "--config" in r.stdout


def test_dataset_merge_coco_help_lists_flags():
    r = runner.invoke(app, ["dataset", "merge", "coco", "--help"])
    assert r.exit_code == 0
    assert "--manifest" in r.stdout
    assert "--out" in r.stdout
    assert "--report-out" in r.stdout


def test_dataset_materialize_images_help_lists_flags():
    r = runner.invoke(app, ["dataset", "materialize-images", "--help"])
    assert r.exit_code == 0
    assert "--coco" in r.stdout
    assert "--out-images-dir" in r.stdout
    assert "--out-coco" in r.stdout
    assert "--merge-manifest" in r.stdout
    assert "--source-images-dir" in r.stdout
    assert "--mode" in r.stdout
    assert "--overwrite" in r.stdout


def test_train_detect_help_lists_smoke_flags():
    r = runner.invoke(app, ["train", "detect", "--help"])
    assert r.exit_code == 0
    assert "--arch" in r.stdout
    assert "--max-steps" in r.stdout
    assert "--limit-train-images" in r.stdout
    assert "--limit-val-images" in r.stdout


def test_train_efficientdet_help_lists_flags():
    r = runner.invoke(app, ["train", "efficientdet", "--help"])
    assert r.exit_code == 0
    assert "--config" in r.stdout
    assert "--variant" in r.stdout
    assert "--max-steps" in r.stdout
    assert "--subset-seed" in r.stdout
    assert "--require-gpu" in r.stdout


def test_eval_detect_help_lists_eval_flags():
    r = runner.invoke(app, ["eval", "detect", "--help"])
    assert r.exit_code == 0
    assert "--run-dir" in r.stdout
    assert "--model" in r.stdout
    assert "--limit-images" in r.stdout
    assert "--score-threshold" in r.stdout


def test_eval_efficientdet_tflite_help_lists_eval_flags():
    r = runner.invoke(app, ["eval", "efficientdet-tflite", "--help"])
    assert r.exit_code == 0
    assert "--coco" in r.stdout
    assert "--images-dir" in r.stdout
    assert "--model" in r.stdout
    assert "--noise-thresholds" in r.stdout
    assert "--num-threads" in r.stdout
    assert "--category-map" in r.stdout


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


def test_golden_detect_help_lists_flags():
    r = runner.invoke(app, ["golden", "detect", "--help"])
    assert r.exit_code == 0
    assert "--model" in r.stdout
    assert "--image" in r.stdout
    assert "--out" in r.stdout
    assert "--max-results" in r.stdout
    assert "--num-threads" in r.stdout
