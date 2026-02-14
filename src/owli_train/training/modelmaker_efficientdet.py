from __future__ import annotations

import csv
import json
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
) -> Path:
    if max_train_images <= 0:
        raise EfficientDetTrainingError("--max-steps requires at least one training image.")

    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    selected_train_images: list[str] = []
    selected_train_set: set[str] = set()
    subset_rows: list[list[str]] = []

    for row in rows:
        if len(row) < 2:
            continue
        set_name = row[0].strip().upper()
        image_name = row[1].strip()
        if set_name.startswith("TRAIN"):
            if image_name in selected_train_set:
                subset_rows.append(row)
                continue
            if len(selected_train_images) < max_train_images:
                selected_train_images.append(image_name)
                selected_train_set.add(image_name)
                subset_rows.append(row)
            continue
        subset_rows.append(row)

    if not selected_train_images:
        raise EfficientDetTrainingError("Model Maker CSV did not contain TRAIN rows.")

    subset_csv.parent.mkdir(parents=True, exist_ok=True)
    with subset_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(subset_rows)

    return subset_csv


def train_efficientdet_from_config(
    *,
    config_path: str | Path,
    variant: str | None = None,
    run_name: str | None = None,
    max_steps: int | None = None,
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

    _, object_detector = ensure_modelmaker_dependencies()
    variant_factory, resolved_variant = _resolve_variant_factory(object_detector, cfg.model.variant)

    run_id, run_dir = _prepare_run_dirs(cfg, run_name)
    artifacts_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs" / "modelmaker"
    logs_dir.mkdir(parents=True, exist_ok=True)

    data_csv_path = cfg.data.csv_path
    effective_epochs = cfg.train.epochs
    effective_batch_size = cfg.train.batch_size
    if max_steps is not None:
        if max_steps <= 0:
            raise EfficientDetTrainingError("--max-steps must be > 0 when provided.")
        effective_epochs = 1
        data_csv_path = _subset_csv_for_max_steps(
            source_csv=cfg.data.csv_path,
            subset_csv=artifacts_dir / "train_subset.csv",
            max_train_images=max_steps * cfg.train.batch_size,
        )

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

    label_map = getattr(train_data, "label_map", {}) or {}
    labels = [str(label_map[key]) for key in sorted(label_map)]
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
