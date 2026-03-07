from pathlib import Path

import pytest

from owli_train.golden.detect import GoldenDetectConfigError, build_golden_detect_config

BASE = {
    "model_path": Path("work/runs/example/artifacts/model.tflite"),
    "image_path": Path("tests/smoke_coco/images/smoke1.jpg"),
    "out_path": Path("work/golden/sample.json"),
    "score_threshold": 0.3,
    "max_results": 20,
}


def test_golden_config_accepts_valid_input():
    cfg = build_golden_detect_config(**BASE)
    assert cfg.model_path.suffix == ".tflite"
    assert cfg.max_results == 20
    assert cfg.num_threads is None


def test_golden_config_rejects_non_tflite_model():
    bad = dict(BASE)
    bad["model_path"] = Path("work/runs/example/artifacts/model.keras")
    with pytest.raises(GoldenDetectConfigError, match=".tflite"):
        build_golden_detect_config(**bad)


def test_golden_config_validates_bounds():
    bad_score = dict(BASE)
    bad_score["score_threshold"] = -0.1
    with pytest.raises(GoldenDetectConfigError, match="score-threshold"):
        build_golden_detect_config(**bad_score)

    bad_max = dict(BASE)
    bad_max["max_results"] = 0
    with pytest.raises(GoldenDetectConfigError, match="max-results"):
        build_golden_detect_config(**bad_max)

    bad_threads = dict(BASE)
    bad_threads["num_threads"] = 0
    with pytest.raises(GoldenDetectConfigError, match="num-threads"):
        build_golden_detect_config(**bad_threads)


def test_golden_config_accepts_num_threads():
    cfg = build_golden_detect_config(**BASE, num_threads=8)
    assert cfg.num_threads == 8
