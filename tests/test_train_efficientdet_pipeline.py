from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from owli_train.training.modelmaker_efficientdet import (
    EfficientDetTrainingError,
    _canonicalize_csv_by_class_order,
    _gpu_missing_error_message,
    _load_label_map_spec,
    _subset_csv_for_max_steps,
    _validate_resolved_label_order,
    _visible_gpu_count,
)


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _read_train_images(path: Path) -> set[str]:
    selected: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            if row[0].strip().upper().startswith("TRAIN"):
                selected.add(row[1].strip())
    return selected


def _read_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def _train_label_sequence(rows: list[list[str]]) -> list[str]:
    labels: list[str] = []
    for row in rows:
        if len(row) < 3:
            continue
        if not row[0].strip().upper().startswith("TRAIN"):
            continue
        label = row[2].strip()
        if label:
            labels.append(label)
    return labels


def test_subset_csv_for_max_steps_is_deterministic_with_seed(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    rows = [["TRAIN", f"img_{idx}.jpg", "person", "0", "0", "", "", "1", "1"] for idx in range(10)]
    rows.extend(
        [
            ["VAL", "val_1.jpg", "car", "0", "0", "", "", "1", "1"],
            ["TEST", "test_1.jpg", "car", "0", "0", "", "", "1", "1"],
        ]
    )
    _write_csv(source, rows)

    out_a = tmp_path / "subset_a.csv"
    out_b = tmp_path / "subset_b.csv"
    out_c = tmp_path / "subset_c.csv"

    _subset_csv_for_max_steps(
        source_csv=source,
        subset_csv=out_a,
        max_train_images=4,
        seed=1337,
    )
    _subset_csv_for_max_steps(
        source_csv=source,
        subset_csv=out_b,
        max_train_images=4,
        seed=1337,
    )
    _subset_csv_for_max_steps(
        source_csv=source,
        subset_csv=out_c,
        max_train_images=4,
        seed=2026,
    )

    assert _read_train_images(out_a) == _read_train_images(out_b)
    assert _read_train_images(out_a) != _read_train_images(out_c)


def test_canonicalize_csv_by_class_order_anchors_first_seen_labels(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source_rows = [
        ["TRAIN", "a.jpg", "car", "0", "0", "", "", "1", "1"],
        ["TRAIN", "b.jpg", "person", "0", "0", "", "", "1", "1"],
        ["TRAIN", "d.jpg", "car", "0", "0", "", "", "1", "1"],
        ["TRAIN", "e.jpg", "person", "0", "0", "", "", "1", "1"],
        ["VAL", "c.jpg", "car", "0", "0", "", "", "1", "1"],
    ]
    _write_csv(source, source_rows)

    result = _canonicalize_csv_by_class_order(
        source_csv=source,
        out_csv=tmp_path / "canonical.csv",
        expected_class_names=["person", "car", "dog"],
    )

    assert result.present_class_names == ["person", "car"]
    assert result.missing_class_names == ["dog"]

    canonical_rows = _read_rows(result.path)
    assert len(canonical_rows) == len(source_rows)

    first_seen: list[str] = []
    seen: set[str] = set()
    for row in canonical_rows:
        if len(row) < 3:
            continue
        label = row[2].strip()
        if label and label not in seen:
            seen.add(label)
            first_seen.append(label)
    assert first_seen == ["person", "car"]

    # Keep all non-anchor rows in original order to avoid long class blocks.
    assert canonical_rows[2:] == source_rows[2:]

    train_labels = _train_label_sequence(canonical_rows)
    transitions = sum(
        1 for idx in range(1, len(train_labels)) if train_labels[idx] != train_labels[idx - 1]
    )
    assert transitions >= 2


def test_canonicalize_csv_by_class_order_rejects_unexpected_labels(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_csv(
        source,
        [
            ["TRAIN", "a.jpg", "car", "0", "0", "", "", "1", "1"],
            ["TRAIN", "b.jpg", "dog", "0", "0", "", "", "1", "1"],
        ],
    )

    with pytest.raises(EfficientDetTrainingError, match="not present in data.label_map_json"):
        _canonicalize_csv_by_class_order(
            source_csv=source,
            out_csv=tmp_path / "canonical.csv",
            expected_class_names=["person", "car"],
        )


def test_load_label_map_spec_validates_schema(tmp_path: Path) -> None:
    valid = tmp_path / "class_names.json"
    valid.write_text(
        json.dumps({"class_names": ["person", "car"], "category_ids": [1, 3]}),
        encoding="utf-8",
    )
    spec = _load_label_map_spec(valid)
    assert spec.class_names == ["person", "car"]
    assert spec.category_ids == [1, 3]

    invalid = tmp_path / "invalid_class_names.json"
    invalid.write_text(
        json.dumps({"class_names": ["person", "person"], "category_ids": [1, 3]}),
        encoding="utf-8",
    )
    with pytest.raises(EfficientDetTrainingError, match="duplicate class names"):
        _load_label_map_spec(invalid)


def test_validate_resolved_label_order_raises_on_mismatch() -> None:
    with pytest.raises(EfficientDetTrainingError, match="class index order mismatch"):
        _validate_resolved_label_order(expected=["person", "car"], actual=["car", "person"])


def test_visible_gpu_count_reads_tf_config() -> None:
    class _Config:
        @staticmethod
        def list_physical_devices(kind: str):
            assert kind == "GPU"
            return ["GPU:0", "GPU:1"]

    class _TF:
        config = _Config()

    assert _visible_gpu_count(_TF()) == 2


def test_gpu_missing_error_message_includes_actionable_hints() -> None:
    class _TF:
        __version__ = "2.8.4"

    message = _gpu_missing_error_message(_TF())
    assert "No TensorFlow GPU device detected" in message
    assert "tensorflow=2.8.4" in message
    assert "MODELMAKER_PYTHON_EXE" in message
