from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

import owli_train.eval.efficientdet_tflite as module
from owli_train.tflite_detect import TFLiteLabelMap, TFLitePreprocessSpec


class _DummyCOCO:
    def __init__(self, _path: str) -> None:
        self.path = _path

    def loadRes(self, _anns: list[dict[str, object]]) -> object:
        raise AssertionError("loadRes must not be called when detections are empty")


class _DummyCOCOEval:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("COCOeval must not be constructed when detections are empty")


def test_eval_handles_empty_detections_without_pycoco_loadres(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image_path)

    coco = {
        "images": [{"id": 1, "file_name": "img.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]}],
        "categories": [{"id": 1, "name": "cat"}],
    }
    coco_path = tmp_path / "instances.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    preprocess = TFLitePreprocessSpec(
        input_size=8,
        input_shape=[1, 8, 8, 3],
        input_dtype="uint8",
        normalization="none",
    )
    runtime = SimpleNamespace(
        preprocess=preprocess,
        builtin_ops_only=True,
        operator_names=["TFLite_Detection_PostProcess"],
        inspect_inputs=[{"name": "input", "shape": [1, 8, 8, 3], "dtype": "uint8"}],
        inspect_outputs=[{"name": "output", "shape": [1, 1, 4], "dtype": "float32"}],
    )

    monkeypatch.setattr(
        module,
        "_ensure_eval_dependencies",
        lambda: (object(), _DummyCOCO, _DummyCOCOEval, object()),
    )
    monkeypatch.setattr(module, "load_tflite_metadata", lambda _path: None)
    monkeypatch.setattr(
        module,
        "load_tflite_label_map",
        lambda _model, _meta: TFLiteLabelMap(class_names=["cat"], source="test"),
    )
    monkeypatch.setattr(module, "create_tflite_runtime", lambda **_kwargs: runtime)
    monkeypatch.setattr(module, "run_tflite_detection", lambda **_kwargs: ([], {}))

    cfg = module.build_eval_efficientdet_tflite_config(
        coco_path=coco_path,
        images_dir=tmp_path,
        model_path=tmp_path / "model.tflite",
        limit_images=1,
        score_threshold=0.3,
        max_detections_per_image=100,
        out_path=tmp_path / "eval.json",
        category_map_path=None,
    )

    artifacts = module.evaluate_efficientdet_tflite(cfg)
    report = json.loads(artifacts.json_report_path.read_text(encoding="utf-8"))

    assert report["num_detections"] == 0
    assert report["metrics"]["AP"] == 0.0
    assert report["noise_metric"]["fp_per_100_images"] == 0.0
    assert artifacts.markdown_report_path.is_file()
