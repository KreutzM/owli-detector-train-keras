from pathlib import Path

import pytest

from owli_train.eval.efficientdet_tflite import (
    EfficientDetTFLiteEvalConfigError,
    _load_category_mapping,
)


def test_load_category_mapping_allows_placeholder_labels() -> None:
    eval_coco = {
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 3, "name": "car"},
        ]
    }
    mapping, source = _load_category_mapping(
        eval_coco=eval_coco,
        label_map=["person", "???", "car", ""],
        category_map_path=None,
    )
    assert source == "label_name_match"
    assert mapping == {0: 1, 2: 3}


def test_load_category_mapping_rejects_unknown_non_placeholder_labels(tmp_path: Path) -> None:
    eval_coco = {
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 3, "name": "car"},
        ]
    }
    with pytest.raises(EfficientDetTFLiteEvalConfigError, match="Could not align class names"):
        _load_category_mapping(
            eval_coco=eval_coco,
            label_map=["person", "totally_unknown_class"],
            category_map_path=None,
        )
