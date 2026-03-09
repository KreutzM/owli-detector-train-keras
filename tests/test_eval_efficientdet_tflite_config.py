from pathlib import Path

import pytest

from owli_train.eval.efficientdet_tflite import (
    EfficientDetTFLiteEvalConfigError,
    build_eval_efficientdet_tflite_config,
)

BASE = {
    "coco_path": Path("tests/smoke_coco/instances_val.json"),
    "images_dir": Path("tests/smoke_coco/images"),
    "model_path": Path("work/runs/example/artifacts/model.tflite"),
    "limit_images": None,
    "score_threshold": 0.3,
    "noise_thresholds": None,
    "max_detections_per_image": 100,
    "num_threads": None,
    "num_workers": None,
    "out_path": None,
    "category_map_path": None,
}


def test_eval_efficientdet_tflite_config_accepts_valid_input():
    cfg = build_eval_efficientdet_tflite_config(**BASE)
    assert cfg.model_path.suffix == ".tflite"
    assert cfg.score_threshold == 0.3
    assert cfg.noise_thresholds == (0.3,)
    assert cfg.num_threads is None
    assert cfg.num_workers is None


def test_eval_efficientdet_tflite_config_rejects_non_tflite():
    bad = dict(BASE)
    bad["model_path"] = Path("work/runs/example/artifacts/model.keras")
    with pytest.raises(EfficientDetTFLiteEvalConfigError, match=".tflite"):
        build_eval_efficientdet_tflite_config(**bad)


def test_eval_efficientdet_tflite_config_validates_numeric_bounds():
    bad_limit = dict(BASE)
    bad_limit["limit_images"] = 0
    with pytest.raises(EfficientDetTFLiteEvalConfigError, match="limit-images"):
        build_eval_efficientdet_tflite_config(**bad_limit)

    bad_score = dict(BASE)
    bad_score["score_threshold"] = 1.2
    with pytest.raises(EfficientDetTFLiteEvalConfigError, match="score-threshold"):
        build_eval_efficientdet_tflite_config(**bad_score)

    bad_noise = dict(BASE)
    bad_noise["noise_thresholds"] = [0.1, 1.5]
    with pytest.raises(EfficientDetTFLiteEvalConfigError, match="noise-thresholds"):
        build_eval_efficientdet_tflite_config(**bad_noise)

    bad_max = dict(BASE)
    bad_max["max_detections_per_image"] = 0
    with pytest.raises(EfficientDetTFLiteEvalConfigError, match="max-detections-per-image"):
        build_eval_efficientdet_tflite_config(**bad_max)

    bad_threads = dict(BASE)
    bad_threads["num_threads"] = 0
    with pytest.raises(EfficientDetTFLiteEvalConfigError, match="num-threads"):
        build_eval_efficientdet_tflite_config(**bad_threads)

    bad_workers = dict(BASE)
    bad_workers["num_workers"] = 0
    with pytest.raises(EfficientDetTFLiteEvalConfigError, match="num-workers"):
        build_eval_efficientdet_tflite_config(**bad_workers)


def test_eval_efficientdet_tflite_config_deduplicates_noise_thresholds():
    cfg = build_eval_efficientdet_tflite_config(
        **{**BASE, "noise_thresholds": [0.05, 0.1, 0.1, 0.3]}
    )
    assert cfg.noise_thresholds == (0.05, 0.1, 0.3)
