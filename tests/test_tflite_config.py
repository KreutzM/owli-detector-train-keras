from pathlib import Path

import pytest

from owli_train.export.tflite_export import (
    TFLiteConfigError,
    build_bench_tflite_config,
    build_export_tflite_config,
    build_inspect_tflite_config,
)


def test_export_config_requires_exactly_one_source():
    with pytest.raises(TFLiteConfigError, match="exactly one model source"):
        build_export_tflite_config(
            run_dir=None,
            saved_model_path=None,
            model_path=None,
            out_path=None,
            quant="none",
            rep_coco=None,
            rep_images_dir=None,
            rep_max_images=8,
            require_builtins_only=False,
        )

    with pytest.raises(TFLiteConfigError, match="exactly one model source"):
        build_export_tflite_config(
            run_dir=Path("work/runs/a"),
            saved_model_path=Path("saved"),
            model_path=None,
            out_path=None,
            quant="none",
            rep_coco=None,
            rep_images_dir=None,
            rep_max_images=8,
            require_builtins_only=False,
        )


def test_export_config_validates_int8_requirements():
    with pytest.raises(TFLiteConfigError, match="requires --rep-coco"):
        build_export_tflite_config(
            run_dir=Path("work/runs/a"),
            saved_model_path=None,
            model_path=None,
            out_path=None,
            quant="int8",
            rep_coco=None,
            rep_images_dir=None,
            rep_max_images=8,
            require_builtins_only=False,
        )


def test_export_config_accepts_fp16_with_run_dir():
    cfg = build_export_tflite_config(
        run_dir=Path("work/runs/a"),
        saved_model_path=None,
        model_path=None,
        out_path=None,
        quant="fp16",
        rep_coco=None,
        rep_images_dir=None,
        rep_max_images=8,
        require_builtins_only=True,
    )

    assert cfg.quant == "fp16"
    assert cfg.run_dir == Path("work/runs/a")
    assert cfg.require_builtins_only is True


def test_bench_config_requires_single_source_and_positive_values():
    with pytest.raises(TFLiteConfigError, match="exactly one source"):
        build_bench_tflite_config(
            run_dir=None,
            model_path=None,
            out_path=None,
            images_dir=None,
            limit_images=8,
            warmup_runs=1,
            runs=8,
        )

    with pytest.raises(TFLiteConfigError, match="limit-images"):
        build_bench_tflite_config(
            run_dir=Path("work/runs/a"),
            model_path=None,
            out_path=None,
            images_dir=None,
            limit_images=0,
            warmup_runs=1,
            runs=8,
        )

    cfg = build_bench_tflite_config(
        run_dir=None,
        model_path=Path("model.tflite"),
        out_path=None,
        images_dir=None,
        limit_images=2,
        warmup_runs=0,
        runs=2,
    )
    assert cfg.model_path == Path("model.tflite")


def test_inspect_config_requires_tflite_model():
    with pytest.raises(TFLiteConfigError, match="\\.tflite"):
        build_inspect_tflite_config(model_path=Path("model.keras"))

    cfg = build_inspect_tflite_config(model_path=Path("detector.tflite"))
    assert cfg.model_path == Path("detector.tflite")
