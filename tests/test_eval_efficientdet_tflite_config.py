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
    "max_detections_per_image": 100,
    "out_path": None,
    "category_map_path": None,
}


def test_eval_efficientdet_tflite_config_accepts_valid_input():
    cfg = build_eval_efficientdet_tflite_config(**BASE)
    assert cfg.model_path.suffix == ".tflite"
    assert cfg.score_threshold == 0.3


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

    bad_max = dict(BASE)
    bad_max["max_detections_per_image"] = 0
    with pytest.raises(EfficientDetTFLiteEvalConfigError, match="max-detections-per-image"):
        build_eval_efficientdet_tflite_config(**bad_max)
