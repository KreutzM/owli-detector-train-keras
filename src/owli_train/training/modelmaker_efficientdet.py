from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


class EfficientDetTrainingError(RuntimeError):
    """Base class for EfficientDet Model Maker training failures."""


class MissingModelMakerDependenciesError(EfficientDetTrainingError):
    """Raised when TensorFlow Lite Model Maker is unavailable."""


@dataclass(frozen=True)
class EfficientDetModelConfig:
    variant: str = "lite2"


@dataclass(frozen=True)
class EfficientDetDataConfig:
    csv_path: Path
    images_dir: Path
    label_map_json: Path | None = None


@dataclass(frozen=True)
class EfficientDetTrainConfig:
    epochs: int = 20
    batch_size: int = 8
    train_whole_model: bool = False


@dataclass(frozen=True)
class EfficientDetOutputsConfig:
    work_dir: Path = Path("work")
    out_dir: Path = Path("outputs")


@dataclass(frozen=True)
class EfficientDetConfig:
    model: EfficientDetModelConfig
    data: EfficientDetDataConfig
    train: EfficientDetTrainConfig
    outputs: EfficientDetOutputsConfig


@dataclass(frozen=True)
class EfficientDetArtifacts:
    run_id: str
    run_dir: Path
    tflite_path: Path
    labels_path: Path
    config_snapshot_path: Path
    mapping_snapshot_path: Path


@dataclass(frozen=True)
class LabelMapSpec:
    class_names: list[str]
    category_ids: list[int] | None


@dataclass(frozen=True)
class CanonicalizedCSV:
    path: Path
    present_class_names: list[str]
    missing_class_names: list[str]


def _as_mapping(obj: Any, label: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"{label} must be an object.")
    return obj


def _as_path(value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty path string.")
    return Path(value)


def _as_optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return Path(value)
    raise ValueError("Optional path values must be non-empty strings when provided.")


def load_efficientdet_config(path: Path) -> EfficientDetConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = _as_mapping(raw, str(path))

    model_raw = _as_mapping(root.get("model", {}), "model")
    model = EfficientDetModelConfig(variant=str(model_raw.get("variant", "lite2")))

    data_raw = _as_mapping(root.get("data", {}), "data")
    data = EfficientDetDataConfig(
        csv_path=_as_path(data_raw.get("csv"), "data.csv"),
        images_dir=_as_path(data_raw.get("images_dir"), "data.images_dir"),
        label_map_json=_as_optional_path(data_raw.get("label_map_json")),
    )

    train_raw = _as_mapping(root.get("train", {}), "train")
    train = EfficientDetTrainConfig(
        epochs=int(train_raw.get("epochs", 20)),
        batch_size=int(train_raw.get("batch_size", 8)),
        train_whole_model=bool(train_raw.get("train_whole_model", False)),
    )
    if train.epochs <= 0:
        raise ValueError("train.epochs must be > 0.")
    if train.batch_size <= 0:
        raise ValueError("train.batch_size must be > 0.")

    outputs_raw = _as_mapping(root.get("outputs", {}), "outputs")
    outputs = EfficientDetOutputsConfig(
        work_dir=_as_optional_path(outputs_raw.get("work_dir")) or Path("work"),
        out_dir=_as_optional_path(outputs_raw.get("out_dir")) or Path("outputs"),
    )

    return EfficientDetConfig(model=model, data=data, train=train, outputs=outputs)


def ensure_modelmaker_dependencies() -> Any:
    try:
        import tensorflow as tf
        from tflite_model_maker import object_detector
    except Exception as exc:  # pragma: no cover - runtime environment specific
        raise MissingModelMakerDependenciesError(
            "TensorFlow Lite Model Maker is required. "
            "Install with: pip install -r requirements\\modelmaker.txt"
        ) from exc
    return tf, object_detector


def _visible_gpu_count(tf_module: Any) -> int:
    try:
        devices = tf_module.config.list_physical_devices("GPU")
    except Exception:
        return 0
    return len(devices)


def _gpu_missing_error_message(tf_module: Any) -> str:
    tf_version = getattr(tf_module, "__version__", "unknown")
    return (
        "No TensorFlow GPU device detected for EfficientDet training. "
        f"tensorflow={tf_version}. "
        "GPU runs on WSL2 require a TensorFlow build/stack that exposes GPUs in this venv. "
        'Check with: python -c "import tensorflow as tf; '
        "print(tf.__version__); print(tf.config.list_physical_devices('GPU'))\". "
        "If empty, fix the ModelMaker venv or set MODELMAKER_PYTHON_EXE to a GPU-ready interpreter."
    )


def _ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise EfficientDetTrainingError(f"{label} was not found: {path}")


def _ensure_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise EfficientDetTrainingError(f"{label} was not found: {path}")


def _run_id(run_name: str | None) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not run_name:
        return stamp
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in run_name)
    cleaned = cleaned.strip("-") or "run"
    return f"{stamp}-{cleaned}"


def _prepare_run_dirs(cfg: EfficientDetConfig, run_name: str | None) -> tuple[str, Path]:
    base = _run_id(run_name)
    run_id = base
    run_dir = cfg.outputs.work_dir / "runs" / run_id
    suffix = 1
    while run_dir.exists():
        run_id = f"{base}-{suffix}"
        run_dir = cfg.outputs.work_dir / "runs" / run_id
        suffix += 1
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def _resolve_variant_factory(object_detector: Any, variant: str) -> Any:
    normalized = variant.lower().replace("efficientdet_", "").replace("efficientdet-", "")
    if normalized.startswith("lite"):
        key = normalized
    elif normalized.startswith("efficientdet_lite"):
        key = normalized.split("efficientdet_")[-1]
    else:
        key = normalized

    variant_map = {
        "lite0": object_detector.EfficientDetLite0Spec,
        "lite1": object_detector.EfficientDetLite1Spec,
        "lite2": object_detector.EfficientDetLite2Spec,
        "lite3": object_detector.EfficientDetLite3Spec,
        "lite4": object_detector.EfficientDetLite4Spec,
    }
    if key not in variant_map:
        raise EfficientDetTrainingError(
            f"Unsupported EfficientDet variant: {variant}. Expected lite0..lite4."
        )
    return variant_map[key], key


def _subset_csv_for_max_steps(
    *,
    source_csv: Path,
    subset_csv: Path,
    max_train_images: int,
    seed: int,
) -> Path:
    if max_train_images <= 0:
        raise EfficientDetTrainingError("--max-steps requires at least one training image.")

    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    train_images: list[str] = []
    train_seen: set[str] = set()
    for row in rows:
        if len(row) < 2:
            continue
        set_name = row[0].strip().upper()
        image_name = row[1].strip()
        if not set_name.startswith("TRAIN"):
            continue
        if image_name in train_seen:
            continue
        train_images.append(image_name)
        train_seen.add(image_name)

    if not train_images:
        raise EfficientDetTrainingError("Model Maker CSV did not contain TRAIN rows.")

    shuffled = list(train_images)
    random.Random(seed).shuffle(shuffled)
    selected_train_set = set(shuffled[:max_train_images])

    subset_rows: list[list[str]] = []
    for row in rows:
        if len(row) < 2:
            continue
        set_name = row[0].strip().upper()
        image_name = row[1].strip()
        if set_name.startswith("TRAIN") and image_name not in selected_train_set:
            continue
        subset_rows.append(row)

    subset_csv.parent.mkdir(parents=True, exist_ok=True)
    with subset_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(subset_rows)

    return subset_csv


def _load_label_map_spec(path: Path) -> LabelMapSpec:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EfficientDetTrainingError(f"Failed to read data.label_map_json: {path}") from exc
    if not isinstance(payload, dict):
        raise EfficientDetTrainingError("data.label_map_json must be a JSON object.")

    raw_names = payload.get("class_names")
    if not isinstance(raw_names, list) or not raw_names:
        raise EfficientDetTrainingError(
            "data.label_map_json must contain a non-empty `class_names` list."
        )
    class_names = [str(item).strip() for item in raw_names]
    if any(not name for name in class_names):
        raise EfficientDetTrainingError("data.label_map_json contains empty class names.")
    if len(set(class_names)) != len(class_names):
        raise EfficientDetTrainingError("data.label_map_json contains duplicate class names.")

    raw_category_ids = payload.get("category_ids")
    if raw_category_ids is None:
        return LabelMapSpec(class_names=class_names, category_ids=None)
    if not isinstance(raw_category_ids, list):
        raise EfficientDetTrainingError("data.label_map_json `category_ids` must be a list.")
    category_ids = [int(item) for item in raw_category_ids]
    if len(category_ids) != len(class_names):
        raise EfficientDetTrainingError(
            "data.label_map_json `category_ids` length must match `class_names` length."
        )
    return LabelMapSpec(class_names=class_names, category_ids=category_ids)


def _canonicalize_csv_by_class_order(
    *,
    source_csv: Path,
    out_csv: Path,
    expected_class_names: list[str],
) -> CanonicalizedCSV:
    if not expected_class_names:
        raise EfficientDetTrainingError("Expected class order must not be empty.")

    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    expected_set = set(expected_class_names)
    first_row_index_by_class: dict[str, int] = {}
    observed_set: set[str] = set()
    for idx, row in enumerate(rows):
        if len(row) < 3:
            continue
        class_name = row[2].strip()
        if not class_name:
            continue
        observed_set.add(class_name)
        if class_name in expected_set and class_name not in first_row_index_by_class:
            first_row_index_by_class[class_name] = idx

    unexpected = sorted(observed_set - expected_set)
    if unexpected:
        preview = ", ".join(unexpected[:8])
        suffix = "" if len(unexpected) <= 8 else ", ..."
        raise EfficientDetTrainingError(
            f"CSV contains classes not present in data.label_map_json: {preview}{suffix}"
        )

    present_class_names = [
        class_name for class_name in expected_class_names if class_name in first_row_index_by_class
    ]
    missing_class_names = [
        class_name
        for class_name in expected_class_names
        if class_name not in first_row_index_by_class
    ]

    # Model Maker assigns label IDs by first-seen class while scanning CSV rows.
    # Move only one anchor row per class to the front to fix label order without
    # turning the full training CSV into large per-class blocks.
    anchor_indices = {first_row_index_by_class[class_name] for class_name in present_class_names}
    canonical_rows = [
        rows[first_row_index_by_class[class_name]] for class_name in present_class_names
    ]
    for idx, row in enumerate(rows):
        if idx in anchor_indices:
            continue
        canonical_rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(canonical_rows)

    return CanonicalizedCSV(
        path=out_csv,
        present_class_names=present_class_names,
        missing_class_names=missing_class_names,
    )


def _validate_resolved_label_order(*, expected: list[str], actual: list[str]) -> None:
    if actual == expected:
        return
    mismatch_at = -1
    for idx, (left, right) in enumerate(zip(expected, actual, strict=False)):
        if left != right:
            mismatch_at = idx
            break
    if mismatch_at < 0 and len(expected) != len(actual):
        mismatch_at = min(len(expected), len(actual))

    expected_preview = ", ".join(expected[:8])
    actual_preview = ", ".join(actual[:8])
    mismatch_msg = f"first mismatch index={mismatch_at}" if mismatch_at >= 0 else "labels differ"
    raise EfficientDetTrainingError(
        "Model Maker class index order mismatch after CSV canonicalization. "
        f"{mismatch_msg}. "
        f"expected=[{expected_preview}] actual=[{actual_preview}]. "
        "This can corrupt pretrained EfficientDet class alignment."
    )


def train_efficientdet_from_config(
    *,
    config_path: str | Path,
    variant: str | None = None,
    run_name: str | None = None,
    max_steps: int | None = None,
    subset_seed: int = 1337,
    require_gpu: bool = False,
) -> EfficientDetArtifacts:
    cfg_path = Path(config_path)
    try:
        cfg = load_efficientdet_config(cfg_path)
    except ValueError as exc:
        raise EfficientDetTrainingError(f"Invalid EfficientDet config: {exc}") from exc
    if variant is not None and variant.strip():
        cfg = EfficientDetConfig(
            model=EfficientDetModelConfig(variant=variant.strip()),
            data=cfg.data,
            train=cfg.train,
            outputs=cfg.outputs,
        )

    _ensure_file(cfg.data.csv_path, "data.csv")
    _ensure_dir(cfg.data.images_dir, "data.images_dir")
    if cfg.data.label_map_json is not None:
        _ensure_file(cfg.data.label_map_json, "data.label_map_json")

    tf_module, object_detector = ensure_modelmaker_dependencies()
    if require_gpu and _visible_gpu_count(tf_module) <= 0:
        raise EfficientDetTrainingError(_gpu_missing_error_message(tf_module))
    variant_factory, resolved_variant = _resolve_variant_factory(object_detector, cfg.model.variant)

    run_id, run_dir = _prepare_run_dirs(cfg, run_name)
    artifacts_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs" / "modelmaker"
    logs_dir.mkdir(parents=True, exist_ok=True)

    data_csv_path = cfg.data.csv_path
    effective_epochs = cfg.train.epochs
    effective_batch_size = cfg.train.batch_size
    if subset_seed < 0:
        raise EfficientDetTrainingError("--subset-seed must be >= 0.")
    if max_steps is not None:
        if max_steps <= 0:
            raise EfficientDetTrainingError("--max-steps must be > 0 when provided.")
        effective_epochs = 1
        data_csv_path = _subset_csv_for_max_steps(
            source_csv=cfg.data.csv_path,
            subset_csv=artifacts_dir / "train_subset.csv",
            max_train_images=max_steps * cfg.train.batch_size,
            seed=subset_seed,
        )

    label_map_spec: LabelMapSpec | None = None
    canonical_csv: CanonicalizedCSV | None = None
    if cfg.data.label_map_json is not None:
        label_map_spec = _load_label_map_spec(cfg.data.label_map_json)
        canonical_csv = _canonicalize_csv_by_class_order(
            source_csv=data_csv_path,
            out_csv=artifacts_dir / "train_canonicalized.csv",
            expected_class_names=label_map_spec.class_names,
        )
        data_csv_path = canonical_csv.path

    try:
        train_data, val_data, _test_data = object_detector.DataLoader.from_csv(
            filename=str(data_csv_path),
            images_dir=str(cfg.data.images_dir),
        )
    except Exception as exc:
        raise EfficientDetTrainingError(
            "Failed to load Model Maker CSV data. Ensure format matches AutoML object-detection CSV."
        ) from exc

    if train_data is None:
        raise EfficientDetTrainingError("Model Maker CSV did not contain TRAIN rows.")

    train_label_map = getattr(train_data, "label_map", {}) or {}
    resolved_labels = [str(train_label_map[key]) for key in sorted(train_label_map)]
    if canonical_csv is not None:
        _validate_resolved_label_order(
            expected=canonical_csv.present_class_names, actual=resolved_labels
        )

    model_spec = variant_factory(
        epochs=effective_epochs,
        batch_size=effective_batch_size,
        model_dir=str(logs_dir),
    )
    try:
        model = object_detector.create(
            train_data=train_data,
            validation_data=val_data,
            model_spec=model_spec,
            epochs=effective_epochs,
            batch_size=effective_batch_size,
            train_whole_model=cfg.train.train_whole_model,
        )
    except Exception as exc:
        raise EfficientDetTrainingError("EfficientDet Model Maker training failed.") from exc

    tflite_path = artifacts_dir / "model.tflite"
    try:
        model.export(
            export_dir=str(artifacts_dir),
            tflite_filename=tflite_path.name,
            with_metadata=True,
            export_metadata_json_file=True,
        )
    except Exception as exc:
        raise EfficientDetTrainingError("Model export to TFLite failed.") from exc

    labels = (
        canonical_csv.present_class_names
        if canonical_csv is not None
        else [str(item) for item in resolved_labels]
    )
    labels_path = artifacts_dir / "labels.txt"
    labels_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")

    class_names_out = artifacts_dir / "class_names.json"
    class_names_out.write_text(
        json.dumps({"class_names": labels}, indent=2),
        encoding="utf-8",
    )

    config_snapshot_path = run_dir / "config.yaml"
    config_snapshot_path.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

    mapping_snapshot_path = run_dir / "mapping_files.json"
    mapping_payload = {
        "variant": resolved_variant,
        "input": {
            "csv": str(cfg.data.csv_path),
            "csv_used_for_training": str(data_csv_path),
            "images_dir": str(cfg.data.images_dir),
            "label_map_json": str(cfg.data.label_map_json) if cfg.data.label_map_json else None,
        },
        "resolved_train": {
            "epochs": effective_epochs,
            "batch_size": effective_batch_size,
            "max_steps": max_steps,
            "subset_seed": subset_seed if max_steps is not None else None,
        },
        "label_alignment": {
            "source": "label_map_json" if label_map_spec is not None else "train_data.label_map",
            "class_count": len(labels),
            "missing_classes_from_training": (
                canonical_csv.missing_class_names if canonical_csv is not None else []
            ),
            "class_index_to_name": labels,
            "class_index_to_category_id": (
                [
                    int(label_map_spec.category_ids[label_map_spec.class_names.index(name)])
                    for name in labels
                ]
                if label_map_spec is not None and label_map_spec.category_ids is not None
                else None
            ),
        },
        "artifacts": {
            "tflite": str(tflite_path),
            "labels": str(labels_path),
            "class_names": str(class_names_out),
        },
    }
    mapping_snapshot_path.write_text(json.dumps(mapping_payload, indent=2), encoding="utf-8")

    if cfg.data.label_map_json is not None:
        shutil.copy2(cfg.data.label_map_json, run_dir / "label_map_input_snapshot.json")

    out_dir = cfg.outputs.out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_dir.txt").write_text(str(run_dir), encoding="utf-8")

    return EfficientDetArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        tflite_path=tflite_path,
        labels_path=labels_path,
        config_snapshot_path=config_snapshot_path,
        mapping_snapshot_path=mapping_snapshot_path,
    )
