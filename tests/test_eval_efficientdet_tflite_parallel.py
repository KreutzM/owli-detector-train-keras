from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

import owli_train.eval.efficientdet_tflite as module
from owli_train.tflite_detect import TFLiteLabelMap, TFLitePreprocessSpec


def test_eval_uses_parallel_collection_when_num_workers_gt_one(tmp_path: Path, monkeypatch) -> None:
    for name in ("img1.jpg", "img2.jpg"):
        Image.new("RGB", (8, 8), color=(0, 0, 0)).save(tmp_path / name)

    coco = {
        "images": [
            {"id": 1, "file_name": "img1.jpg"},
            {"id": 2, "file_name": "img2.jpg"},
        ],
        "annotations": [],
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
        num_threads=1,
        builtin_ops_only=True,
        operator_names=["TFLite_Detection_PostProcess"],
        inspect_inputs=[{"name": "input", "shape": [1, 8, 8, 3], "dtype": "uint8"}],
        inspect_outputs=[{"name": "output", "shape": [1, 1, 4], "dtype": "float32"}],
    )

    class _DummyCOCO:
        def __init__(self, _path: str) -> None:
            self.path = _path

        def loadRes(self, anns: list[dict[str, object]]) -> object:
            return anns

    class _DummyCOCOEval:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.params = SimpleNamespace(imgIds=[], maxDets=[])
            self.stats = [0.0] * 12

        def evaluate(self) -> None:
            return None

        def accumulate(self) -> None:
            return None

        def summarize(self) -> None:
            return None

    called = {"parallel": 0, "serial": 0}

    def fake_parallel(**kwargs):
        called["parallel"] += 1
        assert kwargs["cfg"].num_workers == 2
        assert len(kwargs["eval_images"]) == 2
        return [
            {"image_id": 1, "category_id": 1, "bbox": [0.0, 0.0, 1.0, 1.0], "score": 0.9},
        ]

    def fake_serial(**_kwargs):
        called["serial"] += 1
        raise AssertionError("serial collection must not be used when num_workers > 1")

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
    monkeypatch.setattr(module, "_collect_detections_parallel", fake_parallel)
    monkeypatch.setattr(module, "_collect_detections_serial", fake_serial)

    cfg = module.build_eval_efficientdet_tflite_config(
        coco_path=coco_path,
        images_dir=tmp_path,
        model_path=tmp_path / "model.tflite",
        limit_images=None,
        score_threshold=0.3,
        noise_thresholds=[0.1],
        max_detections_per_image=100,
        num_threads=1,
        num_workers=2,
        out_path=tmp_path / "eval.json",
        category_map_path=None,
    )

    artifacts = module.evaluate_efficientdet_tflite(cfg)
    report = json.loads(artifacts.json_report_path.read_text(encoding="utf-8"))

    assert called == {"parallel": 1, "serial": 0}
    assert report["num_workers"] == 2
    assert report["num_detections"] == 1
