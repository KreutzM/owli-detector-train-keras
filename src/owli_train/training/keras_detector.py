from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from owli_train.data.coco import load_coco, validate_coco
from owli_train.data.split import build_split_coco


class TrainingError(RuntimeError):
    """Base class for detector training failures."""


class MissingTrainingDependenciesError(TrainingError):
    """Raised when TensorFlow / KerasCV are unavailable."""


@dataclass
class ModelConfig:
    arch: str = "yolov8"
    preset: str | None = None
    backbone: str | None = "yolo_v8_xs_backbone"
    input_size: int = 640
    num_classes: int | None = None
    class_names: list[str] = field(default_factory=list)
    bounding_box_format: str = "xywh"


@dataclass
class DataConfig:
    coco_json: Path | None
    images_dir: Path
    splits_json: Path | None = None
    train_coco_json: Path | None = None
    val_coco_json: Path | None = None


@dataclass
class CallbackConfig:
    checkpoint: bool = True
    tensorboard: bool = True


@dataclass
class TrainConfig:
    seed: int = 1337
    batch_size: int = 8
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float | None = None
    steps_per_epoch: int | None = None
    val_steps: int | None = None
    val_frac: float = 0.1
    color_jitter: bool = False
    max_boxes: int = 100
    callbacks: CallbackConfig = field(default_factory=CallbackConfig)


@dataclass
class OutputsConfig:
    work_dir: Path = Path("work")
    out_dir: Path = Path("outputs")


@dataclass
class TrainDetectConfig:
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    outputs: OutputsConfig


@dataclass
class TrainingArtifacts:
    run_id: str
    run_dir: Path
    keras_model_path: Path
    saved_model_dir: Path
    checkpoint_dir: Path
    logs_dir: Path


def _as_mapping(obj: Any, label: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"{label} must be a mapping/object.")
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


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _as_mapping(raw, f"{path}")


def load_train_detect_config(path: str | Path) -> TrainDetectConfig:
    cfg_path = Path(path)
    raw = _load_yaml(cfg_path)

    model_raw = _as_mapping(raw.get("model", {}), "model")
    class_names = [str(v) for v in model_raw.get("class_names", [])]
    parsed_num_classes = model_raw.get("num_classes")
    num_classes = int(parsed_num_classes) if parsed_num_classes is not None else None

    if num_classes is None and class_names:
        num_classes = len(class_names)
    if num_classes is not None and class_names and num_classes != len(class_names):
        raise ValueError("model.num_classes must match len(model.class_names).")

    model = ModelConfig(
        arch=str(model_raw.get("arch", "yolov8")),
        preset=model_raw.get("preset") or model_raw.get("backbone_preset"),
        backbone=model_raw.get("backbone") or "yolo_v8_xs_backbone",
        input_size=int(model_raw.get("input_size", 640)),
        num_classes=num_classes,
        class_names=class_names,
        bounding_box_format=str(model_raw.get("bounding_box_format", "xywh")),
    )

    data_raw = _as_mapping(raw.get("data", {}), "data")
    data = DataConfig(
        coco_json=_as_optional_path(data_raw.get("coco_json")),
        images_dir=_as_path(data_raw.get("images_dir"), "data.images_dir"),
        splits_json=_as_optional_path(data_raw.get("splits_json")),
        train_coco_json=_as_optional_path(data_raw.get("train_coco_json")),
        val_coco_json=_as_optional_path(data_raw.get("val_coco_json")),
    )

    has_separate_train_val = data.train_coco_json is not None and data.val_coco_json is not None
    if not has_separate_train_val and data.coco_json is None:
        raise ValueError(
            "Provide either data.coco_json (optionally with data.splits_json) or both "
            "data.train_coco_json and data.val_coco_json."
        )
    if data.splits_json is not None and data.coco_json is None:
        raise ValueError("data.splits_json requires data.coco_json.")

    train_raw = _as_mapping(raw.get("train", {}), "train")
    callbacks_raw = _as_mapping(train_raw.get("callbacks", {}), "train.callbacks")
    callbacks = CallbackConfig(
        checkpoint=bool(callbacks_raw.get("checkpoint", True)),
        tensorboard=bool(callbacks_raw.get("tensorboard", True)),
    )

    train = TrainConfig(
        seed=int(train_raw.get("seed", 1337)),
        batch_size=int(train_raw.get("batch_size", 8)),
        epochs=int(train_raw.get("epochs", 50)),
        lr=float(train_raw.get("lr", 3e-4)),
        weight_decay=(
            float(train_raw["weight_decay"]) if train_raw.get("weight_decay") is not None else None
        ),
        steps_per_epoch=(
            int(train_raw["steps_per_epoch"])
            if train_raw.get("steps_per_epoch") is not None
            else None
        ),
        val_steps=int(train_raw["val_steps"]) if train_raw.get("val_steps") is not None else None,
        val_frac=float(train_raw.get("val_frac", 0.1)),
        color_jitter=bool(train_raw.get("color_jitter", False)),
        max_boxes=int(train_raw.get("max_boxes", 100)),
        callbacks=callbacks,
    )
    if train.batch_size <= 0:
        raise ValueError("train.batch_size must be > 0.")
    if train.epochs <= 0:
        raise ValueError("train.epochs must be > 0.")
    if train.lr <= 0:
        raise ValueError("train.lr must be > 0.")
    if not (0.0 < train.val_frac < 1.0):
        raise ValueError("train.val_frac must be between 0 and 1.")
    if train.max_boxes <= 0:
        raise ValueError("train.max_boxes must be > 0.")
    if model.input_size <= 0:
        raise ValueError("model.input_size must be > 0.")

    outputs_raw = _as_mapping(raw.get("outputs", {}), "outputs")
    outputs = OutputsConfig(
        work_dir=_as_optional_path(outputs_raw.get("work_dir")) or Path("work"),
        out_dir=_as_optional_path(outputs_raw.get("out_dir")) or Path("outputs"),
    )

    return TrainDetectConfig(model=model, data=data, train=train, outputs=outputs)


def ensure_training_dependencies() -> tuple[Any, Any]:
    try:
        import keras_cv
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - runtime environment specific
        raise MissingTrainingDependenciesError(
            "TensorFlow and KerasCV are required for `train detect`. "
            "Install with: pip install -r requirements\\keras.txt"
        ) from exc
    return tf, keras_cv


def _ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise TrainingError(
            f"{label} was not found: {path}. Check your config paths before training."
        )


def _ensure_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise TrainingError(
            f"{label} was not found: {path}. Check your config paths before training."
        )


def _validate_dataset_paths(cfg: TrainDetectConfig) -> None:
    _ensure_dir(cfg.data.images_dir, "data.images_dir")

    if cfg.data.coco_json is not None:
        _ensure_file(cfg.data.coco_json, "data.coco_json")
    if cfg.data.train_coco_json is not None:
        _ensure_file(cfg.data.train_coco_json, "data.train_coco_json")
    if cfg.data.val_coco_json is not None:
        _ensure_file(cfg.data.val_coco_json, "data.val_coco_json")
    if cfg.data.splits_json is not None:
        _ensure_file(cfg.data.splits_json, "data.splits_json")


def _load_splits(path: Path) -> dict[str, list[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    obj = _as_mapping(raw, "splits_json")
    for key in ("train", "val"):
        if key not in obj or not isinstance(obj[key], list):
            raise ValueError(f"splits_json missing list key: {key}")
    return {
        "train": [int(v) for v in obj["train"]],
        "val": [int(v) for v in obj["val"]],
        "test": [int(v) for v in obj.get("test", [])],
    }


def _make_train_val_split(
    image_ids: list[int],
    seed: int,
    val_frac: float,
) -> dict[str, list[int]]:
    ids = sorted(int(v) for v in image_ids)
    rnd = random.Random(seed)
    rnd.shuffle(ids)

    if len(ids) == 1:
        return {"train": ids, "val": []}

    n_val = max(1, int(len(ids) * val_frac))
    n_val = min(n_val, len(ids) - 1)
    return {"train": ids[n_val:], "val": ids[:n_val], "test": []}


def _load_train_val_coco(cfg: TrainDetectConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    images_dir = cfg.data.images_dir

    if cfg.data.train_coco_json and cfg.data.val_coco_json:
        train_coco = load_coco(cfg.data.train_coco_json)
        val_coco = load_coco(cfg.data.val_coco_json)
        validate_coco(train_coco, images_dir=images_dir)
        validate_coco(val_coco, images_dir=images_dir)
        return train_coco, val_coco

    assert cfg.data.coco_json is not None
    base_coco = load_coco(cfg.data.coco_json)
    validate_coco(base_coco, images_dir=images_dir)

    if cfg.data.splits_json is not None:
        splits = _load_splits(cfg.data.splits_json)
    else:
        splits = _make_train_val_split(
            image_ids=[int(img["id"]) for img in base_coco["images"]],
            seed=cfg.train.seed,
            val_frac=cfg.train.val_frac,
        )

    train_coco = build_split_coco(base_coco, splits["train"])
    val_coco = build_split_coco(base_coco, splits["val"])
    return train_coco, val_coco


def _resolve_class_names(
    configured: list[str],
    train_coco: dict[str, Any],
    val_coco: dict[str, Any],
) -> list[str]:
    if configured:
        return configured

    names_by_id: dict[int, str] = {}
    for source in (train_coco, val_coco):
        for cat in source["categories"]:
            names_by_id[int(cat["id"])] = str(cat["name"])
    return [names_by_id[k] for k in sorted(names_by_id)]


def _build_category_id_to_class(
    categories: list[dict[str, Any]],
    class_names: list[str],
) -> dict[int, int]:
    class_name_to_index = {name: idx for idx, name in enumerate(class_names)}
    mapping: dict[int, int] = {}
    for category in categories:
        cat_name = str(category["name"])
        if cat_name in class_name_to_index:
            mapping[int(category["id"])] = class_name_to_index[cat_name]
    return mapping


def _build_image_records(
    coco: dict[str, Any],
    images_dir: Path,
    category_to_class: dict[int, int],
    limit_images: int | None = None,
) -> list[dict[str, Any]]:
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        image_id = int(ann["image_id"])
        annotations_by_image.setdefault(image_id, []).append(ann)

    images_sorted = sorted(coco["images"], key=lambda img: int(img["id"]))
    if limit_images is not None:
        images_sorted = images_sorted[:limit_images]

    records: list[dict[str, Any]] = []
    for image in images_sorted:
        image_id = int(image["id"])
        boxes: list[list[float]] = []
        classes: list[int] = []

        for ann in annotations_by_image.get(image_id, []):
            cat_id = int(ann["category_id"])
            if cat_id not in category_to_class:
                continue
            bbox = [float(v) for v in ann["bbox"]]
            boxes.append(bbox)
            classes.append(category_to_class[cat_id])

        records.append(
            {
                "image_id": image_id,
                "image_path": str(images_dir / str(image["file_name"])),
                "boxes": boxes,
                "classes": classes,
            }
        )

    return records


def _build_augment_fn(tf: Any, keras_cv: Any, cfg: TrainDetectConfig, training: bool) -> Any:
    fmt = cfg.model.bounding_box_format
    target_size = (cfg.model.input_size, cfg.model.input_size)

    if training:
        flip = keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format=fmt)
        jittered_resize = keras_cv.layers.JitteredResize(
            target_size=target_size,
            scale_factor=(0.8, 1.25),
            bounding_box_format=fmt,
        )

        def augment(sample: dict[str, Any]) -> dict[str, Any]:
            sample = flip(sample)
            sample = jittered_resize(sample)
            if cfg.train.color_jitter:
                images = sample["images"]
                images = tf.image.random_brightness(images, max_delta=20.0)
                images = tf.image.random_contrast(images, lower=0.8, upper=1.2)
                images = tf.clip_by_value(images, 0.0, 255.0)
                sample["images"] = images
            return sample

        return augment

    resize = keras_cv.layers.Resizing(
        height=cfg.model.input_size,
        width=cfg.model.input_size,
        bounding_box_format=fmt,
        pad_to_aspect_ratio=True,
    )

    def preprocess(sample: dict[str, Any]) -> dict[str, Any]:
        return resize(sample)

    return preprocess


def _build_tf_dataset(
    tf: Any,
    keras_cv: Any,
    records: list[dict[str, Any]],
    cfg: TrainDetectConfig,
    training: bool,
) -> Any:
    if not records:
        raise TrainingError("No records available for dataset creation.")

    output_signature = {
        "image_path": tf.TensorSpec(shape=(), dtype=tf.string),
        "boxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        "classes": tf.TensorSpec(shape=(None,), dtype=tf.float32),
    }

    def gen() -> Any:
        for rec in records:
            yield {
                "image_path": rec["image_path"],
                "boxes": np.asarray(rec["boxes"], dtype=np.float32),
                "classes": np.asarray(rec["classes"], dtype=np.float32),
            }

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    if training:
        ds = ds.shuffle(max(1, len(records)), seed=cfg.train.seed, reshuffle_each_iteration=True)

    def decode(item: dict[str, Any]) -> dict[str, Any]:
        image_bytes = tf.io.read_file(item["image_path"])
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        return {
            "images": image,
            "bounding_boxes": {
                "boxes": item["boxes"],
                "classes": item["classes"],
            },
        }

    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        _build_augment_fn(tf=tf, keras_cv=keras_cv, cfg=cfg, training=training),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.ragged_batch(cfg.train.batch_size, drop_remainder=False)

    max_boxes = max(cfg.train.max_boxes, max(len(rec["boxes"]) for rec in records))

    def densify(sample: dict[str, Any]) -> dict[str, Any]:
        boxes = keras_cv.bounding_box.to_dense(sample["bounding_boxes"], max_boxes=max_boxes)
        return {"images": tf.cast(sample["images"], tf.float32), "bounding_boxes": boxes}

    ds = ds.map(densify, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def _build_model(tf: Any, keras_cv: Any, cfg: TrainDetectConfig) -> Any:
    arch = cfg.model.arch.lower()
    if arch != "yolov8":
        raise TrainingError(f"Unsupported model.arch: {cfg.model.arch}")

    num_classes = cfg.model.num_classes or len(cfg.model.class_names)
    if num_classes <= 0:
        raise TrainingError("num_classes must be positive.")

    if cfg.model.preset:
        model = keras_cv.models.YOLOV8Detector.from_preset(
            cfg.model.preset,
            bounding_box_format=cfg.model.bounding_box_format,
            num_classes=num_classes,
        )
    else:
        backbone = keras_cv.models.YOLOV8Backbone.from_preset(
            cfg.model.backbone or "yolo_v8_xs_backbone",
            load_weights=False,
        )
        model = keras_cv.models.YOLOV8Detector(
            num_classes=num_classes,
            bounding_box_format=cfg.model.bounding_box_format,
            backbone=backbone,
        )

    if cfg.train.weight_decay is not None:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.train.lr)

    model.compile(
        classification_loss="binary_crossentropy",
        box_loss="ciou",
        optimizer=optimizer,
    )
    return model


def _run_id_from_name(run_name: str | None) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not run_name:
        return stamp

    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in run_name)
    cleaned = cleaned.strip("-") or "run"
    return f"{stamp}-{cleaned}"


def _prepare_run_dirs(cfg: TrainDetectConfig, run_name: str | None) -> tuple[str, Path, Path, Path]:
    base_run_id = _run_id_from_name(run_name)
    run_id = base_run_id
    run_dir = cfg.outputs.work_dir / "runs" / run_id
    suffix = 1
    while run_dir.exists():
        run_id = f"{base_run_id}-{suffix}"
        run_dir = cfg.outputs.work_dir / "runs" / run_id
        suffix += 1
    logs_dir = run_dir / "logs"
    ckpt_dir = run_dir / "checkpoints"
    artifacts_dir = run_dir / "artifacts"

    logs_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir.mkdir(parents=True, exist_ok=False)
    artifacts_dir.mkdir(parents=True, exist_ok=False)

    return run_id, run_dir, logs_dir, ckpt_dir


def _build_callbacks(
    tf: Any, cfg: TrainDetectConfig, logs_dir: Path, checkpoint_dir: Path
) -> list[Any]:
    callbacks: list[Any] = [
        tf.keras.callbacks.CSVLogger(str(logs_dir / "train.csv"), append=False),
    ]

    if cfg.train.callbacks.tensorboard:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(logs_dir / "tensorboard")))

    if cfg.train.callbacks.checkpoint:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_dir / "epoch-{epoch:03d}.weights.h5"),
                save_weights_only=True,
                save_best_only=False,
            )
        )

    return callbacks


def _compute_steps(
    cfg: TrainDetectConfig, train_count: int, val_count: int, max_steps: int | None
) -> tuple[int, int, int]:
    steps_per_epoch = cfg.train.steps_per_epoch or max(1, ceil(train_count / cfg.train.batch_size))
    val_steps = cfg.train.val_steps or max(1, ceil(max(1, val_count) / cfg.train.batch_size))
    epochs = cfg.train.epochs

    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0 when provided.")
        steps_per_epoch = min(steps_per_epoch, max_steps)
        epochs = 1

    return epochs, steps_per_epoch, val_steps


def _save_snapshots(
    config_path: Path,
    cfg: TrainDetectConfig,
    run_id: str,
    run_dir: Path,
    class_names: list[str],
    category_mapping: dict[int, int],
) -> None:
    (run_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    snapshot = {
        "run_id": run_id,
        "class_names": class_names,
        "category_id_to_class_index": {str(k): int(v) for k, v in sorted(category_mapping.items())},
        "bounding_box_format": cfg.model.bounding_box_format,
    }
    (run_dir / "label_map_snapshot.json").write_text(
        json.dumps(snapshot, indent=2),
        encoding="utf-8",
    )


def train_detector_from_config(
    config_path: str | Path,
    run_name: str | None = None,
    max_steps: int | None = None,
    limit_train_images: int | None = None,
    limit_val_images: int | None = None,
    resume: str | Path | None = None,
) -> TrainingArtifacts:
    cfg_path = Path(config_path)
    cfg = load_train_detect_config(cfg_path)
    _validate_dataset_paths(cfg)
    tf, keras_cv = ensure_training_dependencies()

    run_id, run_dir, logs_dir, checkpoint_dir = _prepare_run_dirs(cfg, run_name)

    train_coco, val_coco = _load_train_val_coco(cfg)
    class_names = _resolve_class_names(cfg.model.class_names, train_coco, val_coco)
    if cfg.model.num_classes is not None and cfg.model.num_classes != len(class_names):
        raise TrainingError("Configured num_classes does not match resolved class_names.")
    cfg.model.class_names = class_names
    cfg.model.num_classes = len(class_names)

    category_mapping = _build_category_id_to_class(train_coco["categories"], class_names)
    _save_snapshots(cfg_path, cfg, run_id, run_dir, class_names, category_mapping)

    train_records = _build_image_records(
        train_coco,
        images_dir=cfg.data.images_dir,
        category_to_class=category_mapping,
        limit_images=limit_train_images,
    )
    val_records = _build_image_records(
        val_coco,
        images_dir=cfg.data.images_dir,
        category_to_class=category_mapping,
        limit_images=limit_val_images,
    )

    if not train_records:
        raise TrainingError("No training images resolved after filters.")
    if not val_records:
        val_records = train_records[: min(1, len(train_records))]

    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    tf.random.set_seed(cfg.train.seed)

    train_ds = _build_tf_dataset(
        tf=tf, keras_cv=keras_cv, records=train_records, cfg=cfg, training=True
    )
    val_ds = _build_tf_dataset(
        tf=tf, keras_cv=keras_cv, records=val_records, cfg=cfg, training=False
    )

    model = _build_model(tf=tf, keras_cv=keras_cv, cfg=cfg)

    if resume is not None:
        model.load_weights(str(Path(resume)))

    epochs, steps_per_epoch, val_steps = _compute_steps(
        cfg=cfg,
        train_count=len(train_records),
        val_count=len(val_records),
        max_steps=max_steps,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=_build_callbacks(
            tf=tf, cfg=cfg, logs_dir=logs_dir, checkpoint_dir=checkpoint_dir
        ),
        verbose=1,
    )

    artifacts_dir = run_dir / "artifacts"
    keras_model_path = artifacts_dir / "detector.keras"
    saved_model_dir = artifacts_dir / "saved_model"

    model.save(str(keras_model_path))
    if hasattr(model, "export"):
        model.export(str(saved_model_dir))
    else:  # pragma: no cover - depends on TensorFlow/Keras runtime version
        tf.saved_model.save(model, str(saved_model_dir))

    out_dir = cfg.outputs.out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_dir.txt").write_text(str(run_dir), encoding="utf-8")

    return TrainingArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        keras_model_path=keras_model_path,
        saved_model_dir=saved_model_dir,
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
    )
