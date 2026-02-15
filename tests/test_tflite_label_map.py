from pathlib import Path

from owli_train import tflite_detect
from owli_train.tflite_detect import load_tflite_label_map


def test_load_tflite_label_map_reads_class_names_json_dict(tmp_path: Path) -> None:
    model = tmp_path / "model.tflite"
    model.write_bytes(b"fake")

    (tmp_path / "class_names.json").write_text(
        '{"class_names": ["person", "car"]}',
        encoding="utf-8",
    )

    labels = load_tflite_label_map(model, metadata=None)
    assert labels.class_names == ["person", "car"]
    assert labels.source == "class_names.json"


def test_load_tflite_label_map_uses_embedded_metadata_when_sidecars_missing(
    tmp_path: Path, monkeypatch
) -> None:
    model = tmp_path / "model.tflite"
    model.write_bytes(b"fake")

    monkeypatch.setattr(
        tflite_detect,
        "_load_embedded_label_map",
        lambda _model_path: (["person", "???", "car"], "labelmap.txt"),
    )

    labels = load_tflite_label_map(model, metadata=None)
    assert labels.class_names == ["person", "???", "car"]
    assert labels.source == "tflite_metadata:labelmap.txt"
