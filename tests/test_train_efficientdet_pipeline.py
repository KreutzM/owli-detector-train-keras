from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from owli_train.training.modelmaker_efficientdet import (
    EfficientDetAugmentationConfig,
    EfficientDetTrainingError,
    _build_model_spec_hparams,
    _canonicalize_csv_by_class_order,
    _collect_train_split_class_coverage,
    _enforce_train_split_class_contract,
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


def test_collect_train_split_class_coverage_uses_train_rows_only(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_csv(
        source,
        [
            ["TRAIN", "a.jpg", "person", "0", "0", "", "", "1", "1"],
            ["VAL", "b.jpg", "bus", "0", "0", "", "", "1", "1"],
            ["TEST", "c.jpg", "car", "0", "0", "", "", "1", "1"],
        ],
    )

    coverage = _collect_train_split_class_coverage(
        source_csv=source,
        expected_class_names=["person", "bus", "car"],
    )

    assert coverage.present_class_names == ["person"]
    assert coverage.missing_class_names == ["bus", "car"]


def test_enforce_train_split_class_contract_allows_complete_train_split(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_csv(
        source,
        [
            ["TRAIN", "a.jpg", "person", "0", "0", "", "", "1", "1"],
            ["TRAIN", "b.jpg", "car", "0", "0", "", "", "1", "1"],
            ["TRAIN", "c.jpg", "bus", "0", "0", "", "", "1", "1"],
        ],
    )

    coverage = _enforce_train_split_class_contract(
        source_csv=source,
        expected_class_names=["person", "car", "bus"],
        allow_missing_train_classes=False,
    )

    assert coverage.present_class_names == ["person", "car", "bus"]
    assert coverage.missing_class_names == []


def test_enforce_train_split_class_contract_raises_for_missing_expected_class(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    _write_csv(
        source,
        [
            ["TRAIN", "a.jpg", "person", "0", "0", "", "", "1", "1"],
            ["TRAIN", "b.jpg", "car", "0", "0", "", "", "1", "1"],
        ],
    )

    with pytest.raises(EfficientDetTrainingError) as exc_info:
        _enforce_train_split_class_contract(
            source_csv=source,
            expected_class_names=["person", "car", "bus"],
            allow_missing_train_classes=False,
        )
    message = str(exc_info.value)
    assert "missing from TRAIN rows in data.csv" in message
    assert "bus" in message
    assert "data.label_map_json" in message
    assert "labels.txt/class_names.json" in message
    assert "train.allow_missing_train_classes=true" in message


def test_enforce_train_split_class_contract_allows_explicit_override(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_csv(
        source,
        [
            ["TRAIN", "a.jpg", "person", "0", "0", "", "", "1", "1"],
            ["TRAIN", "b.jpg", "car", "0", "0", "", "", "1", "1"],
        ],
    )

    coverage = _enforce_train_split_class_contract(
        source_csv=source,
        expected_class_names=["person", "car", "bus"],
        allow_missing_train_classes=True,
    )

    assert coverage.present_class_names == ["person", "car"]
    assert coverage.missing_class_names == ["bus"]


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


def test_build_model_spec_hparams_serializes_supported_augmentation_fields() -> None:
    hparams = _build_model_spec_hparams(
        EfficientDetAugmentationConfig(
            rand_hflip=False,
            jitter_min=0.8,
            jitter_max=1.2,
            autoaugment_policy="v2",
        )
    )

    assert hparams == ("input_rand_hflip=false,jitter_min=0.8,jitter_max=1.2,autoaugment_policy=v2")


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


def test_train_efficientdet_from_config_passes_augmentation_hparams_to_model_spec(
    tmp_path: Path, monkeypatch
) -> None:
    from owli_train.training import modelmaker_efficientdet as mm

    cfg_path = tmp_path / "effdet.yaml"
    dataset_csv = tmp_path / "dataset.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_csv(
        dataset_csv,
        [["TRAIN", "a.jpg", "person", "0", "0", "", "", "1", "1"]],
    )
    cfg_path.write_text(
        f"""
model:
  variant: lite2
data:
  csv: {dataset_csv}
  images_dir: {images_dir}
train:
  epochs: 1
  batch_size: 1
  augmentation:
    rand_hflip: false
    jitter_min: 0.8
    jitter_max: 1.2
    autoaugment_policy: v1
outputs:
  work_dir: {tmp_path / "work"}
  out_dir: {tmp_path / "outputs"}
""",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeTrainData:
        label_map = {1: "person"}

    class FakeDataLoader:
        @staticmethod
        def from_csv(*, filename: str, images_dir: str):
            captured["from_csv"] = {"filename": filename, "images_dir": images_dir}
            return FakeTrainData(), None, None

    class FakeModel:
        def export(
            self,
            *,
            export_dir: str,
            tflite_filename: str,
            with_metadata: bool,
            export_metadata_json_file: bool,
        ) -> None:
            captured["export"] = {
                "export_dir": export_dir,
                "tflite_filename": tflite_filename,
                "with_metadata": with_metadata,
                "export_metadata_json_file": export_metadata_json_file,
            }
            Path(export_dir, tflite_filename).write_text("fake-model", encoding="utf-8")

    class FakeObjectDetector:
        DataLoader = FakeDataLoader

        @staticmethod
        def create(**kwargs):
            captured["create"] = kwargs
            return FakeModel()

    class FakeTF:
        class config:
            @staticmethod
            def list_physical_devices(_kind: str):
                return []

    def fake_variant_factory(**kwargs):
        captured["variant_kwargs"] = kwargs
        return "fake-model-spec"

    monkeypatch.setattr(mm, "ensure_modelmaker_dependencies", lambda: (FakeTF, FakeObjectDetector))
    monkeypatch.setattr(
        mm,
        "_resolve_variant_factory",
        lambda _object_detector, _variant: (fake_variant_factory, "lite2"),
    )

    artifacts = mm.train_efficientdet_from_config(config_path=cfg_path)

    assert artifacts.tflite_path.name == "model.tflite"
    assert captured["variant_kwargs"] == {
        "epochs": 1,
        "batch_size": 1,
        "hparams": "input_rand_hflip=false,jitter_min=0.8,jitter_max=1.2,autoaugment_policy=v1",
        "model_dir": str(artifacts.run_dir / "logs" / "modelmaker"),
    }

    mapping_payload = json.loads(artifacts.mapping_snapshot_path.read_text(encoding="utf-8"))
    assert mapping_payload["augmentation"] == {
        "config": {
            "rand_hflip": False,
            "jitter_min": 0.8,
            "jitter_max": 1.2,
            "autoaugment_policy": "v1",
        },
        "model_spec_hparams": "input_rand_hflip=false,jitter_min=0.8,jitter_max=1.2,autoaugment_policy=v1",
        "applied_via": "model_spec.hparams",
    }
