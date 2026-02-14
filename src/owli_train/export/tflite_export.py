from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from owli_train.data.coco import load_coco, validate_coco


class TFLiteExportError(RuntimeError):
    """Base class for export/bench failures."""


class TFLiteConfigError(TFLiteExportError):
    """Raised when CLI/config inputs are invalid."""


class MissingTFLiteDependenciesError(TFLiteExportError):
    """Raised when TensorFlow is unavailable."""


@dataclass(frozen=True)
class ExportTFLiteConfig:
    run_dir: Path | None
    saved_model_path: Path | None
    model_path: Path | None
    out_path: Path | None
    quant: str
    rep_coco: Path | None
    rep_images_dir: Path | None
    rep_max_images: int
    require_builtins_only: bool


@dataclass(frozen=True)
class ExportTFLiteArtifacts:
    model_path: Path
    tflite_path: Path
    metadata_path: Path
    quant: str
    source_type: str
    builtin_ops_only: bool | None = None


@dataclass(frozen=True)
class BenchTFLiteConfig:
    run_dir: Path | None
    model_path: Path | None
    out_path: Path | None
    images_dir: Path | None
    limit_images: int
    warmup_runs: int
    runs: int


@dataclass(frozen=True)
class BenchTFLiteArtifacts:
    model_path: Path
    report_path: Path


@dataclass(frozen=True)
class InspectTFLiteConfig:
    model_path: Path


@dataclass(frozen=True)
class InspectTFLiteArtifacts:
    model_path: Path
    builtin_ops_only: bool
    operator_names: list[str]
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]


def ensure_tflite_dependencies() -> Any:
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - runtime environment specific
        raise MissingTFLiteDependenciesError(
            "TensorFlow is required. Install with: pip install -r requirements\\keras.txt"
        ) from exc
    return tf


def _require_positive_int(value: int, flag_name: str) -> None:
    if value <= 0:
        raise TFLiteConfigError(f"{flag_name} must be > 0.")


def _resolve_exactly_one_source(
    *, run_dir: Path | None, saved_model: Path | None, model: Path | None
) -> None:
    provided = [v is not None for v in (run_dir, saved_model, model)]
    if sum(provided) != 1:
        raise TFLiteConfigError(
            "Provide exactly one model source: --run-dir OR --saved-model OR --model."
        )


def build_export_tflite_config(
    *,
    run_dir: Path | None,
    saved_model_path: Path | None,
    model_path: Path | None,
    out_path: Path | None,
    quant: str,
    rep_coco: Path | None,
    rep_images_dir: Path | None,
    rep_max_images: int,
    require_builtins_only: bool,
) -> ExportTFLiteConfig:
    _resolve_exactly_one_source(run_dir=run_dir, saved_model=saved_model_path, model=model_path)

    normalized_quant = quant.lower().strip()
    if normalized_quant not in {"none", "fp16", "int8"}:
        raise TFLiteConfigError("--quant must be one of: none, fp16, int8.")

    _require_positive_int(rep_max_images, "--rep-max-images")

    if normalized_quant == "int8":
        if rep_coco is None or rep_images_dir is None:
            raise TFLiteConfigError(
                "--quant int8 requires --rep-coco and --rep-images-dir for calibration."
            )

    return ExportTFLiteConfig(
        run_dir=Path(run_dir) if run_dir is not None else None,
        saved_model_path=Path(saved_model_path) if saved_model_path is not None else None,
        model_path=Path(model_path) if model_path is not None else None,
        out_path=Path(out_path) if out_path is not None else None,
        quant=normalized_quant,
        rep_coco=Path(rep_coco) if rep_coco is not None else None,
        rep_images_dir=Path(rep_images_dir) if rep_images_dir is not None else None,
        rep_max_images=rep_max_images,
        require_builtins_only=require_builtins_only,
    )


def build_inspect_tflite_config(*, model_path: Path) -> InspectTFLiteConfig:
    resolved = Path(model_path)
    if resolved.suffix.lower() != ".tflite":
        raise TFLiteConfigError("--model must point to a .tflite file.")
    return InspectTFLiteConfig(model_path=resolved)


def build_bench_tflite_config(
    *,
    run_dir: Path | None,
    model_path: Path | None,
    out_path: Path | None,
    images_dir: Path | None,
    limit_images: int,
    warmup_runs: int,
    runs: int,
) -> BenchTFLiteConfig:
    provided = [run_dir is not None, model_path is not None]
    if sum(provided) != 1:
        raise TFLiteConfigError("Provide exactly one source: --run-dir OR --model.")

    _require_positive_int(limit_images, "--limit-images")
    if warmup_runs < 0:
        raise TFLiteConfigError("--warmup-runs must be >= 0.")
    _require_positive_int(runs, "--runs")

    return BenchTFLiteConfig(
        run_dir=Path(run_dir) if run_dir is not None else None,
        model_path=Path(model_path) if model_path is not None else None,
        out_path=Path(out_path) if out_path is not None else None,
        images_dir=Path(images_dir) if images_dir is not None else None,
        limit_images=limit_images,
        warmup_runs=warmup_runs,
        runs=runs,
    )


def _ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise TFLiteConfigError(f"{label} was not found: {path}")


def _ensure_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise TFLiteConfigError(f"{label} was not found: {path}")


def _read_run_snapshot(run_dir: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}

    label_snapshot = run_dir / "label_map_snapshot.json"
    if label_snapshot.is_file():
        payload = json.loads(label_snapshot.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            snapshot.update(payload)

    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if isinstance(payload, dict):
            model_cfg = payload.get("model") or {}
            if isinstance(model_cfg, dict):
                snapshot.setdefault("class_names", model_cfg.get("class_names") or [])
                snapshot.setdefault("bounding_box_format", model_cfg.get("bounding_box_format"))
                snapshot.setdefault("input_size", model_cfg.get("input_size"))

    return snapshot


def _resolve_export_source(cfg: ExportTFLiteConfig) -> tuple[str, Path, dict[str, Any]]:
    if cfg.run_dir is not None:
        _ensure_dir(cfg.run_dir, "--run-dir")
        saved_model = cfg.run_dir / "artifacts" / "saved_model"
        keras_model = cfg.run_dir / "artifacts" / "detector.keras"
        snapshot = _read_run_snapshot(cfg.run_dir)
        if saved_model.is_dir():
            return "saved_model", saved_model, snapshot
        if keras_model.is_file():
            return "keras", keras_model, snapshot
        raise TFLiteConfigError(
            f"No model artifact found in run dir. Expected {saved_model} or {keras_model}"
        )

    if cfg.saved_model_path is not None:
        _ensure_dir(cfg.saved_model_path, "--saved-model")
        return "saved_model", cfg.saved_model_path, {}

    assert cfg.model_path is not None
    _ensure_file(cfg.model_path, "--model")
    if cfg.model_path.suffix.lower() != ".keras":
        raise TFLiteConfigError("--model must point to a .keras file.")
    return "keras", cfg.model_path, {}


def _default_export_out(cfg: ExportTFLiteConfig) -> Path:
    if cfg.out_path is not None:
        return cfg.out_path
    if cfg.run_dir is not None:
        return cfg.run_dir / "artifacts" / "detector.tflite"
    return Path("outputs/model.tflite")


def _infer_input_size_from_model(model: Any) -> int:
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, tuple) and len(shape) >= 3:
        h = shape[1]
        w = shape[2]
        if isinstance(h, int) and h > 0 and isinstance(w, int) and w > 0:
            return max(h, w)
    return 640


def _letterbox_and_normalize(image_path: Path, input_size: int) -> np.ndarray:
    with Image.open(image_path) as img:
        image = img.convert("RGB")

    orig_w, orig_h = image.size
    scale = min(input_size / float(orig_w), input_size / float(orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (input_size, input_size), color=(0, 0, 0))

    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr


def _representative_data(
    *,
    cfg: ExportTFLiteConfig,
    input_size: int,
) -> list[np.ndarray]:
    assert cfg.rep_coco is not None
    assert cfg.rep_images_dir is not None

    _ensure_file(cfg.rep_coco, "--rep-coco")
    _ensure_dir(cfg.rep_images_dir, "--rep-images-dir")

    coco = load_coco(cfg.rep_coco)
    validate_coco(coco, images_dir=cfg.rep_images_dir)

    images = sorted(coco["images"], key=lambda item: int(item["id"]))
    selected = images[: cfg.rep_max_images]

    if not selected:
        raise TFLiteConfigError("Representative dataset resolved to zero images.")

    samples: list[np.ndarray] = []
    for image in selected:
        image_path = cfg.rep_images_dir / str(image["file_name"])
        tensor = _letterbox_and_normalize(image_path, input_size=input_size)
        samples.append(np.expand_dims(tensor, axis=0).astype(np.float32))
    return samples


def _configure_converter_quantization(
    *,
    converter: Any,
    tf: Any,
    cfg: ExportTFLiteConfig,
    input_size: int,
) -> None:
    if cfg.quant == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        return

    if cfg.quant == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        rep_samples = _representative_data(cfg=cfg, input_size=input_size)

        def representative_dataset() -> Any:
            for sample in rep_samples:
                yield [sample]

        converter.representative_dataset = representative_dataset


def _convert_from_saved_model(
    *,
    tf: Any,
    saved_model_path: Path,
    cfg: ExportTFLiteConfig,
    input_size: int,
) -> bytes:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
    _configure_converter_quantization(converter=converter, tf=tf, cfg=cfg, input_size=input_size)
    return converter.convert()


def _convert_from_keras_model(
    *,
    tf: Any,
    keras_path: Path,
    cfg: ExportTFLiteConfig,
    input_size: int | None,
) -> tuple[bytes, int]:
    keras_model = tf.keras.models.load_model(str(keras_path), compile=False)
    inferred_input_size = _infer_input_size_from_model(keras_model)
    resolved_input_size = int(input_size) if input_size is not None else inferred_input_size

    # Export through a fixed-input wrapper to avoid dynamic-shape converter fallbacks.
    try:
        wrapped_input = tf.keras.Input(
            shape=(resolved_input_size, resolved_input_size, 3), dtype=tf.float32
        )
        wrapped_output = keras_model(wrapped_input)
    except ValueError:
        # Fallback for fixed-shape models when requested input size is incompatible.
        resolved_input_size = inferred_input_size
        wrapped_input = tf.keras.Input(
            shape=(resolved_input_size, resolved_input_size, 3), dtype=tf.float32
        )
        wrapped_output = keras_model(wrapped_input)
    wrapped_model = tf.keras.Model(wrapped_input, wrapped_output)

    converter = tf.lite.TFLiteConverter.from_keras_model(wrapped_model)
    _configure_converter_quantization(
        converter=converter, tf=tf, cfg=cfg, input_size=resolved_input_size
    )
    return converter.convert(), resolved_input_size


def export_tflite(cfg: ExportTFLiteConfig) -> ExportTFLiteArtifacts:
    tf = ensure_tflite_dependencies()

    source_type, source_path, snapshot = _resolve_export_source(cfg)
    out_path = _default_export_out(cfg)
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")

    snapshot_input = snapshot.get("input_size")
    preferred_input_size = (
        int(snapshot_input)
        if isinstance(snapshot_input, (int, float)) and int(snapshot_input) > 0
        else None
    )
    input_size = preferred_input_size or 640

    try:
        if source_type == "saved_model":
            tflite_bytes = _convert_from_saved_model(
                tf=tf,
                saved_model_path=source_path,
                cfg=cfg,
                input_size=input_size,
            )
        else:
            tflite_bytes, input_size = _convert_from_keras_model(
                tf=tf,
                keras_path=source_path,
                cfg=cfg,
                input_size=preferred_input_size,
            )
    except Exception as exc:  # pragma: no cover - runtime/environment specific
        if source_type == "saved_model" and cfg.run_dir is not None:
            fallback_keras = cfg.run_dir / "artifacts" / "detector.keras"
            if fallback_keras.is_file():
                try:
                    tflite_bytes, input_size = _convert_from_keras_model(
                        tf=tf,
                        keras_path=fallback_keras,
                        cfg=cfg,
                        input_size=preferred_input_size,
                    )
                    source_type = "keras_fallback"
                    source_path = fallback_keras
                except Exception as fallback_exc:  # pragma: no cover
                    raise TFLiteExportError(
                        "TFLite conversion failed for SavedModel and .keras fallback."
                    ) from fallback_exc
            else:
                raise TFLiteExportError(
                    "TFLite conversion failed for SavedModel, and no .keras fallback was found."
                ) from exc
        else:
            raise TFLiteExportError(
                "TFLite conversion failed. Try --model <detector.keras> or --quant none."
            ) from exc

    builtin_ops_only, operator_names, _inputs, _outputs = _inspect_tflite_model(
        tf=tf, model_content=tflite_bytes
    )
    if cfg.require_builtins_only and not builtin_ops_only:
        raise TFLiteExportError(
            "Export produced a model requiring SELECT_TF_OPS (Flex), but "
            "--require-builtins-only was set. Try --quant none/fp16, export from "
            "--model <detector.keras>, or use a different model architecture."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_bytes)

    class_names = snapshot.get("class_names") or []
    if not isinstance(class_names, list):
        class_names = []

    metadata = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_type": source_type,
        "source_path": str(source_path),
        "run_dir": str(cfg.run_dir) if cfg.run_dir is not None else None,
        "quant": cfg.quant,
        "input_size": input_size,
        "bbox_format": snapshot.get("bounding_box_format") or "xywh",
        "class_names": [str(v) for v in class_names],
        "settings": {
            "rep_coco": str(cfg.rep_coco) if cfg.rep_coco is not None else None,
            "rep_images_dir": str(cfg.rep_images_dir) if cfg.rep_images_dir is not None else None,
            "rep_max_images": cfg.rep_max_images,
            "require_builtins_only": cfg.require_builtins_only,
        },
        "android_compat": {
            "builtin_ops_only": builtin_ops_only,
            "requires_select_tf_ops": not builtin_ops_only,
            "operator_names": operator_names,
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return ExportTFLiteArtifacts(
        model_path=source_path,
        tflite_path=out_path,
        metadata_path=meta_path,
        quant=cfg.quant,
        source_type=source_type,
        builtin_ops_only=builtin_ops_only,
    )


def _resolve_bench_model_path(cfg: BenchTFLiteConfig) -> Path:
    if cfg.run_dir is not None:
        _ensure_dir(cfg.run_dir, "--run-dir")
        model_path = cfg.run_dir / "artifacts" / "detector.tflite"
        _ensure_file(model_path, "detector.tflite in --run-dir/artifacts")
        return model_path

    assert cfg.model_path is not None
    _ensure_file(cfg.model_path, "--model")
    if cfg.model_path.suffix.lower() != ".tflite":
        raise TFLiteConfigError("--model must point to a .tflite file.")
    return cfg.model_path


def _default_bench_out(cfg: BenchTFLiteConfig) -> Path:
    if cfg.out_path is not None:
        return cfg.out_path
    if cfg.run_dir is not None:
        return cfg.run_dir / "reports" / "bench_tflite.json"
    return Path("work/reports/bench_tflite.json")


def _load_input_size_from_meta(model_path: Path, fallback: int) -> int:
    meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")
    if not meta_path.is_file():
        return fallback

    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return fallback

    value = payload.get("input_size") if isinstance(payload, dict) else None
    if isinstance(value, int) and value > 0:
        return value
    return fallback


def _dtype_np_from_tf(dtype: Any) -> Any:
    if hasattr(dtype, "as_numpy_dtype"):
        return dtype.as_numpy_dtype
    return np.float32


def _summarize_tensor_details(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for item in details:
        shape_values = item.get("shape") or []
        shape = [int(v) for v in shape_values]
        summary.append(
            {
                "name": str(item.get("name")),
                "shape": shape,
                "dtype": str(item.get("dtype")),
            }
        )
    return summary


def _extract_operator_names(interpreter: Any) -> list[str]:
    getter = getattr(interpreter, "_get_ops_details", None)
    if getter is None:
        return []

    try:
        op_details = getter()
    except Exception:
        return []

    names: list[str] = []
    seen: set[str] = set()
    for item in op_details:
        op_name = item.get("op_name") or item.get("name")
        if not op_name:
            continue
        name = str(op_name)
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _is_builtin_ops_only(operator_names: list[str]) -> bool:
    return not any(name.startswith("Flex") for name in operator_names)


def _inspect_tflite_interpreter(
    interpreter: Any,
) -> tuple[bool, list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    operator_names = _extract_operator_names(interpreter)
    builtin_ops_only = _is_builtin_ops_only(operator_names)

    try:
        input_details = _summarize_tensor_details(interpreter.get_input_details())
    except Exception:
        input_details = []

    try:
        output_details = _summarize_tensor_details(interpreter.get_output_details())
    except Exception:
        output_details = []

    return builtin_ops_only, operator_names, input_details, output_details


def _inspect_tflite_model(
    *,
    tf: Any,
    model_path: Path | None = None,
    model_content: bytes | None = None,
) -> tuple[bool, list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    if model_path is None and model_content is None:
        raise TFLiteConfigError("Internal error: expected model_path or model_content.")

    if model_path is not None:
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
    else:
        interpreter = tf.lite.Interpreter(model_content=model_content)

    return _inspect_tflite_interpreter(interpreter)


def _build_bench_inputs(
    *,
    cfg: BenchTFLiteConfig,
    input_shape: list[int],
    input_dtype: Any,
) -> list[np.ndarray]:
    h = int(input_shape[1])
    w = int(input_shape[2])
    c = int(input_shape[3])

    if cfg.images_dir is None:
        if input_dtype == np.uint8:
            return [
                np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
                for _ in range(cfg.limit_images)
            ]
        return [
            np.random.random(size=input_shape).astype(np.float32) for _ in range(cfg.limit_images)
        ]

    _ensure_dir(cfg.images_dir, "--images-dir")
    image_files = sorted(
        [
            p
            for p in cfg.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )

    if not image_files:
        raise TFLiteConfigError("--images-dir does not contain any supported image files.")

    selected = image_files[: cfg.limit_images]
    tensors: list[np.ndarray] = []
    for image_path in selected:
        arr = _letterbox_and_normalize(image_path, input_size=max(h, w))
        if c == 1:
            arr = np.mean(arr, axis=2, keepdims=True)

        tensor = np.expand_dims(arr.astype(np.float32), axis=0)
        if input_dtype == np.uint8:
            tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
        else:
            tensor = tensor.astype(np.float32)
        tensors.append(tensor)

    return tensors


def bench_tflite(cfg: BenchTFLiteConfig) -> BenchTFLiteArtifacts:
    tf = ensure_tflite_dependencies()

    model_path = _resolve_bench_model_path(cfg)
    out_path = _default_bench_out(cfg)

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    input_details = interpreter.get_input_details()
    if not input_details:
        raise TFLiteExportError("TFLite model has no input tensors.")

    input_info = input_details[0]
    shape = [int(v) for v in input_info["shape"]]
    if len(shape) != 4:
        raise TFLiteExportError(f"Expected 4D input tensor for image model, got shape={shape}.")

    input_size = _load_input_size_from_meta(model_path, fallback=640)
    if shape[1] <= 0 or shape[2] <= 0:
        shape[1] = input_size
        shape[2] = input_size
    if shape[0] <= 0:
        shape[0] = 1
    if shape[3] <= 0:
        shape[3] = 3

    interpreter.resize_tensor_input(int(input_info["index"]), shape)
    try:
        interpreter.allocate_tensors()
    except RuntimeError as exc:  # pragma: no cover - runtime/environment specific
        raise TFLiteExportError(
            "Failed to initialize TFLite interpreter. If this model uses SELECT_TF_OPS, "
            "you may need TensorFlow Lite Flex delegate support."
        ) from exc

    refreshed_input_info = interpreter.get_input_details()[0]
    input_dtype = _dtype_np_from_tf(refreshed_input_info["dtype"])

    samples = _build_bench_inputs(
        cfg=cfg,
        input_shape=[int(v) for v in refreshed_input_info["shape"]],
        input_dtype=input_dtype,
    )

    for _ in range(cfg.warmup_runs):
        sample = samples[_ % len(samples)]
        interpreter.set_tensor(int(refreshed_input_info["index"]), sample)
        interpreter.invoke()

    latencies_ms: list[float] = []
    for i in range(cfg.runs):
        sample = samples[i % len(samples)]
        t0 = time.perf_counter()
        interpreter.set_tensor(int(refreshed_input_info["index"]), sample)
        interpreter.invoke()
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)

    output_details = interpreter.get_output_details()
    outputs_summary = [
        {
            "name": str(item.get("name")),
            "shape": [int(v) for v in item.get("shape", [])],
            "dtype": str(item.get("dtype")),
        }
        for item in output_details
    ]

    report = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model_path": str(model_path),
        "run_dir": str(cfg.run_dir) if cfg.run_dir is not None else None,
        "images_dir": str(cfg.images_dir) if cfg.images_dir is not None else None,
        "limit_images": cfg.limit_images,
        "warmup_runs": cfg.warmup_runs,
        "runs": cfg.runs,
        "input": {
            "shape": [int(v) for v in refreshed_input_info["shape"]],
            "dtype": str(refreshed_input_info["dtype"]),
        },
        "outputs": outputs_summary,
        "latency_ms": {
            "mean": float(statistics.mean(latencies_ms)),
            "median": float(statistics.median(latencies_ms)),
            "min": float(min(latencies_ms)),
            "max": float(max(latencies_ms)),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return BenchTFLiteArtifacts(model_path=model_path, report_path=out_path)


def inspect_tflite(cfg: InspectTFLiteConfig) -> InspectTFLiteArtifacts:
    tf = ensure_tflite_dependencies()

    _ensure_file(cfg.model_path, "--model")
    if cfg.model_path.suffix.lower() != ".tflite":
        raise TFLiteConfigError("--model must point to a .tflite file.")

    builtin_ops_only, operator_names, inputs, outputs = _inspect_tflite_model(
        tf=tf, model_path=cfg.model_path
    )
    return InspectTFLiteArtifacts(
        model_path=cfg.model_path,
        builtin_ops_only=builtin_ops_only,
        operator_names=operator_names,
        inputs=inputs,
        outputs=outputs,
    )


def export_saved_model_to_tflite(saved_model_dir: str | Path, out_path: str | Path) -> None:
    """Backward-compatible wrapper for direct SavedModel export (no quantization)."""

    cfg = build_export_tflite_config(
        run_dir=None,
        saved_model_path=Path(saved_model_dir),
        model_path=None,
        out_path=Path(out_path),
        quant="none",
        rep_coco=None,
        rep_images_dir=None,
        rep_max_images=32,
        require_builtins_only=False,
    )
    _ = export_tflite(cfg)
