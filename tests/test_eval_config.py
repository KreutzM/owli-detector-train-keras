from pathlib import Path

import pytest

from owli_train.eval.detect import EvalConfigError, build_eval_detect_config

BASE = {
    "coco_path": Path("tests/data/coco_min.json"),
    "images_dir": Path("tests/data"),
    "run_dir": Path("work/runs/example"),
    "model_path": None,
    "limit_images": None,
    "score_threshold": 0.25,
    "max_detections_per_image": 100,
    "out_path": None,
    "category_map_path": None,
}


def test_eval_config_accepts_run_dir_source():
    cfg = build_eval_detect_config(**BASE)
    assert cfg.run_dir == Path("work/runs/example")
    assert cfg.model_path is None


def test_eval_config_requires_single_model_source():
    both = dict(BASE)
    both["model_path"] = Path("outputs/detector.keras")
    with pytest.raises(EvalConfigError, match="not both"):
        build_eval_detect_config(**both)

    missing = dict(BASE)
    missing["run_dir"] = None
    with pytest.raises(EvalConfigError, match="Provide either"):
        build_eval_detect_config(**missing)


def test_eval_config_validates_numeric_bounds():
    bad_limit = dict(BASE)
    bad_limit["limit_images"] = 0
    with pytest.raises(EvalConfigError, match="limit-images"):
        build_eval_detect_config(**bad_limit)

    bad_score = dict(BASE)
    bad_score["score_threshold"] = 1.5
    with pytest.raises(EvalConfigError, match="score-threshold"):
        build_eval_detect_config(**bad_score)

    bad_max_det = dict(BASE)
    bad_max_det["max_detections_per_image"] = 0
    with pytest.raises(EvalConfigError, match="max-detections-per-image"):
        build_eval_detect_config(**bad_max_det)
