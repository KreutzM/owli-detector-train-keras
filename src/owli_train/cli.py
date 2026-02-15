from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich import print

from owli_train.data.coco import (
    load_coco,
    load_label_map,
    normalize_coco,
    validate_coco,
    write_coco,
)
from owli_train.data.modelmaker_csv import export_coco_to_modelmaker_csv
from owli_train.data.split import split_coco_image_ids, write_split_coco_files, write_splits
from owli_train.data.yolo_adapter import import_yolo_to_coco
from owli_train.eval.detect import (
    EvalConfigError,
    MissingEvalDependenciesError,
    build_eval_detect_config,
    evaluate_detect,
)
from owli_train.eval.efficientdet_tflite import (
    EfficientDetTFLiteEvalConfigError,
    MissingEfficientDetTFLiteEvalDependenciesError,
    build_eval_efficientdet_tflite_config,
    evaluate_efficientdet_tflite,
)
from owli_train.export.tflite_export import (
    MissingTFLiteDependenciesError,
    TFLiteConfigError,
    TFLiteExportError,
    build_bench_tflite_config,
    build_export_tflite_config,
    build_inspect_tflite_config,
)
from owli_train.export.tflite_export import (
    bench_tflite as run_bench_tflite,
)
from owli_train.export.tflite_export import (
    export_tflite as run_export_tflite,
)
from owli_train.export.tflite_export import (
    inspect_tflite as run_inspect_tflite,
)
from owli_train.golden.detect import (
    GoldenDetectConfigError,
    MissingGoldenDependenciesError,
    build_golden_detect_config,
    generate_golden_detect,
)
from owli_train.training.keras_detector import (
    MissingTrainingDependenciesError,
    TrainingError,
    train_detector_from_config,
)
from owli_train.training.modelmaker_efficientdet import (
    EfficientDetTrainingError,
    MissingModelMakerDependenciesError,
    train_efficientdet_from_config,
)

DEFAULT_NORMALIZED_OUT = Path("work/normalized/instances.json")
DEFAULT_SPLITS_OUT_DIR = Path("work/splits")
DEFAULT_MODELMAKER_CSV_OUT = Path("work/datasets/modelmaker/dataset.csv")

app = typer.Typer(add_completion=False, no_args_is_help=True)
dataset_app = typer.Typer(no_args_is_help=True)
dataset_import_app = typer.Typer(no_args_is_help=True)
dataset_export_app = typer.Typer(no_args_is_help=True)
train_app = typer.Typer(no_args_is_help=True)
eval_app = typer.Typer(no_args_is_help=True)
export_app = typer.Typer(no_args_is_help=True)
bench_app = typer.Typer(no_args_is_help=True)
inspect_app = typer.Typer(no_args_is_help=True)
golden_app = typer.Typer(no_args_is_help=True)

app.add_typer(dataset_app, name="dataset")
dataset_app.add_typer(dataset_import_app, name="import")
dataset_app.add_typer(dataset_export_app, name="export")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(export_app, name="export")
app.add_typer(bench_app, name="bench")
app.add_typer(inspect_app, name="inspect")
app.add_typer(golden_app, name="golden")


def _delegate_to_modelmaker_python(args: list[str]) -> int | None:
    requested = os.environ.get("MODELMAKER_PYTHON_EXE")
    if not requested:
        return None
    if os.environ.get("OWLI_MODELMAKER_DELEGATED") == "1":
        return None

    current_exe = str(Path(sys.executable).resolve())
    requested_path = Path(requested)
    if requested_path.exists():
        resolved_requested = str(requested_path.resolve())
        if resolved_requested == current_exe:
            return None

    env = os.environ.copy()
    env["OWLI_MODELMAKER_DELEGATED"] = "1"
    cmd = [requested, "-m", "owli_train", *args]
    try:
        result = subprocess.run(cmd, env=env, check=False)
    except FileNotFoundError:
        print(
            "[red]ERROR[/red] MODELMAKER_PYTHON_EXE is set, but the executable was not found: "
            f"{requested}"
        )
        return 1
    return int(result.returncode)


@dataset_app.command("validate")
def dataset_validate(
    coco: Annotated[Path, typer.Option("--coco", exists=True, readable=True)],
    images_dir: Annotated[
        Path | None,
        typer.Option(
            "--images-dir",
            file_okay=False,
            dir_okay=True,
            exists=True,
            readable=True,
        ),
    ] = None,
):
    obj = load_coco(coco)
    s = validate_coco(obj, images_dir=images_dir)
    print(f"[green]OK[/green] COCO: images={s.images}, ann={s.annotations}, cats={s.categories}")


@dataset_app.command("normalize")
def dataset_normalize(
    coco: Annotated[Path, typer.Option("--coco", exists=True, readable=True)],
    out: Annotated[Path, typer.Option("--out")] = DEFAULT_NORMALIZED_OUT,
    images_dir: Annotated[
        Path | None,
        typer.Option(
            "--images-dir",
            file_okay=False,
            dir_okay=True,
            exists=True,
            readable=True,
        ),
    ] = None,
    label_map: Annotated[
        Path | None, typer.Option("--label-map", exists=True, readable=True)
    ] = None,
):
    obj = load_coco(coco)
    _ = validate_coco(obj, images_dir=images_dir)

    mapping = load_label_map(label_map) if label_map is not None else None
    normalized = normalize_coco(obj, label_map=mapping)
    _ = validate_coco(normalized, images_dir=images_dir)

    out_path = write_coco(out, normalized)
    print(f"[green]OK[/green] wrote normalized COCO: {out_path}")


@dataset_app.command("summarize")
def dataset_summarize(
    coco: Annotated[Path, typer.Option("--coco", exists=True, readable=True)],
):
    obj = load_coco(coco)
    s = validate_coco(obj)
    cats = ", ".join(s.category_names[:20])
    more = "" if len(s.category_names) <= 20 else f" (+{len(s.category_names) - 20} more)"
    print(f"images: {s.images}")
    print(f"annotations: {s.annotations}")
    print(f"categories: {s.categories}")
    print(f"category names: {cats}{more}")


@dataset_app.command("split")
def dataset_split(
    coco: Annotated[Path, typer.Option("--coco", exists=True, readable=True)],
    out_dir: Annotated[Path, typer.Option("--out-dir")] = DEFAULT_SPLITS_OUT_DIR,
    seed: Annotated[int, typer.Option("--seed")] = 1337,
    train_frac: Annotated[float, typer.Option("--train-frac")] = 0.8,
    val_frac: Annotated[float, typer.Option("--val-frac")] = 0.1,
    write_coco_files: Annotated[bool, typer.Option("--write-coco")] = False,
):
    obj = load_coco(coco)
    _ = validate_coco(obj)
    splits = split_coco_image_ids(obj, seed=seed, train_frac=train_frac, val_frac=val_frac)
    out = write_splits(out_dir, splits)

    if write_coco_files:
        written = write_split_coco_files(out_dir=out_dir, coco=obj, splits=splits)
        print(
            "[green]OK[/green] wrote "
            f"{out} and split COCO files: train={written['train']}, val={written['val']}, test={written['test']}"
        )
    else:
        print(f"[green]OK[/green] wrote {out}")


@dataset_import_app.command("yolo")
def dataset_import_yolo(
    yolo_dir: Annotated[
        Path,
        typer.Option(
            "--yolo-dir",
            exists=True,
            readable=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    out: Annotated[Path | None, typer.Option("--out")] = None,
    data_yaml: Annotated[
        Path | None, typer.Option("--data-yaml", exists=True, readable=True)
    ] = None,
):
    out_path = (
        out if out is not None else Path("work") / "datasets" / yolo_dir.name / "instances.json"
    )
    try:
        artifacts = import_yolo_to_coco(
            yolo_dir=yolo_dir,
            out_path=out_path,
            data_yaml=data_yaml,
        )
    except ValueError as exc:
        print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    print(f"[green]OK[/green] wrote COCO: {artifacts.coco_path}")
    print(f"class_names_json: {artifacts.class_names_path}")
    print(
        "summary: "
        f"images={artifacts.images}, annotations={artifacts.annotations}, categories={artifacts.categories}"
    )


@dataset_export_app.command("modelmaker-csv")
def dataset_export_modelmaker_csv(
    coco: Annotated[Path, typer.Option("--coco", exists=True, readable=True)],
    images_dir: Annotated[
        Path,
        typer.Option(
            "--images-dir",
            exists=True,
            readable=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    out: Annotated[Path, typer.Option("--out")] = DEFAULT_MODELMAKER_CSV_OUT,
    splits_json: Annotated[
        Path | None, typer.Option("--splits-json", exists=True, readable=True)
    ] = None,
    class_names_out: Annotated[Path | None, typer.Option("--class-names-out")] = None,
):
    try:
        artifacts = export_coco_to_modelmaker_csv(
            coco_path=coco,
            images_dir=images_dir,
            out_csv=out,
            splits_json=splits_json,
            class_names_out=class_names_out,
        )
    except ValueError as exc:
        print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    print(f"[green]OK[/green] wrote CSV: {artifacts.csv_path}")
    print(f"class_names_json: {artifacts.class_names_path}")
    print(
        "summary: "
        f"rows={artifacts.rows}, images={artifacts.images}, annotations={artifacts.annotations}"
    )


@train_app.command("detect")
def train_detect(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True)],
    run_name: Annotated[str | None, typer.Option("--run-name")] = None,
    arch: Annotated[str | None, typer.Option("--arch")] = None,
    max_steps: Annotated[int | None, typer.Option("--max-steps")] = None,
    limit_train_images: Annotated[int | None, typer.Option("--limit-train-images")] = None,
    limit_val_images: Annotated[int | None, typer.Option("--limit-val-images")] = None,
    resume: Annotated[Path | None, typer.Option("--resume", exists=True, readable=True)] = None,
):
    try:
        artifacts = train_detector_from_config(
            config_path=config,
            run_name=run_name,
            arch=arch,
            max_steps=max_steps,
            limit_train_images=limit_train_images,
            limit_val_images=limit_val_images,
            resume=resume,
        )
    except (MissingTrainingDependenciesError, TrainingError) as exc:
        print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    print(f"[green]OK[/green] run={artifacts.run_id}")
    print(f"run_dir: {artifacts.run_dir}")
    print(f"keras_model: {artifacts.keras_model_path}")
    print(f"saved_model: {artifacts.saved_model_dir}")


@train_app.command("efficientdet")
def train_efficientdet(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True)],
    variant: Annotated[str | None, typer.Option("--variant")] = None,
    run_name: Annotated[str | None, typer.Option("--run-name")] = None,
    max_steps: Annotated[int | None, typer.Option("--max-steps")] = None,
):
    try:
        artifacts = train_efficientdet_from_config(
            config_path=config,
            variant=variant,
            run_name=run_name,
            max_steps=max_steps,
        )
    except (EfficientDetTrainingError, MissingModelMakerDependenciesError) as exc:
        print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    print(f"[green]OK[/green] run={artifacts.run_id}")
    print(f"run_dir: {artifacts.run_dir}")
    print(f"tflite: {artifacts.tflite_path}")
    print(f"labels: {artifacts.labels_path}")


@eval_app.command("detect")
def eval_detect_cli(
    coco: Annotated[Path, typer.Option("--coco", exists=True, readable=True)],
    images_dir: Annotated[
        Path,
        typer.Option(
            "--images-dir",
            exists=True,
            readable=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    run_dir: Annotated[
        Path | None,
        typer.Option(
            "--run-dir",
            exists=True,
            readable=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
    model: Annotated[Path | None, typer.Option("--model", exists=True, readable=True)] = None,
    limit_images: Annotated[int | None, typer.Option("--limit-images")] = None,
    score_threshold: Annotated[float, typer.Option("--score-threshold")] = 0.25,
    max_detections_per_image: Annotated[int, typer.Option("--max-detections-per-image")] = 100,
    category_map: Annotated[
        Path | None, typer.Option("--category-map", exists=True, readable=True)
    ] = None,
    out: Annotated[Path | None, typer.Option("--out")] = None,
):
    try:
        cfg = build_eval_detect_config(
            coco_path=coco,
            images_dir=images_dir,
            run_dir=run_dir,
            model_path=model,
            limit_images=limit_images,
            score_threshold=score_threshold,
            max_detections_per_image=max_detections_per_image,
            out_path=out,
            category_map_path=category_map,
        )
        artifacts = evaluate_detect(cfg)
    except (EvalConfigError, MissingEvalDependenciesError) as exc:
        print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    print(f"[green]OK[/green] model={artifacts.model_path}")
    print(f"report_json: {artifacts.json_report_path}")
    print(f"report_md: {artifacts.markdown_report_path}")


@eval_app.command("efficientdet-tflite")
def eval_efficientdet_tflite_cli(
    coco: Annotated[Path, typer.Option("--coco", exists=True, readable=True)],
    images_dir: Annotated[
        Path,
        typer.Option(
            "--images-dir",
            exists=True,
            readable=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    model: Annotated[Path, typer.Option("--model", exists=True, readable=True)],
    limit_images: Annotated[int | None, typer.Option("--limit-images")] = None,
    score_threshold: Annotated[float, typer.Option("--score-threshold")] = 0.3,
    noise_thresholds: Annotated[str | None, typer.Option("--noise-thresholds")] = None,
    max_detections_per_image: Annotated[int, typer.Option("--max-detections-per-image")] = 100,
    category_map: Annotated[
        Path | None, typer.Option("--category-map", exists=True, readable=True)
    ] = None,
    out: Annotated[Path | None, typer.Option("--out")] = None,
):
    delegate_args = [
        "eval",
        "efficientdet-tflite",
        "--coco",
        str(coco),
        "--images-dir",
        str(images_dir),
        "--model",
        str(model),
        "--score-threshold",
        str(score_threshold),
        "--max-detections-per-image",
        str(max_detections_per_image),
    ]
    if noise_thresholds is not None:
        delegate_args.extend(["--noise-thresholds", noise_thresholds])
    if limit_images is not None:
        delegate_args.extend(["--limit-images", str(limit_images)])
    if category_map is not None:
        delegate_args.extend(["--category-map", str(category_map)])
    if out is not None:
        delegate_args.extend(["--out", str(out)])

    delegated = _delegate_to_modelmaker_python(delegate_args)
    if delegated is not None:
        raise typer.Exit(code=delegated)

    parsed_noise_thresholds: list[float] | None = None
    if noise_thresholds is not None:
        parsed_noise_thresholds = []
        for value in noise_thresholds.split(","):
            raw = value.strip()
            if not raw:
                continue
            try:
                parsed_noise_thresholds.append(float(raw))
            except ValueError as exc:
                print("[red]ERROR[/red] --noise-thresholds expects comma-separated numeric values.")
                raise typer.Exit(code=1) from exc

    try:
        cfg = build_eval_efficientdet_tflite_config(
            coco_path=coco,
            images_dir=images_dir,
            model_path=model,
            limit_images=limit_images,
            score_threshold=score_threshold,
            noise_thresholds=parsed_noise_thresholds,
            max_detections_per_image=max_detections_per_image,
            out_path=out,
            category_map_path=category_map,
        )
        artifacts = evaluate_efficientdet_tflite(cfg)
    except (
        EfficientDetTFLiteEvalConfigError,
        MissingEfficientDetTFLiteEvalDependenciesError,
    ) as exc:
        print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    print(f"[green]OK[/green] model={artifacts.model_path}")
    print(f"report_json: {artifacts.json_report_path}")
    print(f"report_md: {artifacts.markdown_report_path}")


@export_app.command("tflite")
def export_tflite(
    run_dir: Annotated[
        Path | None,
        typer.Option(
            "--run-dir",
            exists=True,
            readable=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
    saved_model: Annotated[
        Path | None,
        typer.Option(
            "--saved-model",
            exists=True,
            readable=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
    model: Annotated[Path | None, typer.Option("--model", exists=True, readable=True)] = None,
    out: Annotated[Path | None, typer.Option("--out")] = None,
    quant: Annotated[str, typer.Option("--quant")] = "none",
    rep_coco: Annotated[Path | None, typer.Option("--rep-coco", exists=True, readable=True)] = None,
    rep_images_dir: Annotated[
        Path | None,
        typer.Option(
            "--rep-images-dir",
            exists=True,
            readable=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
    rep_max_images: Annotated[int, typer.Option("--rep-max-images")] = 32,
    require_builtins_only: Annotated[bool, typer.Option("--require-builtins-only")] = False,
):
    try:
        cfg = build_export_tflite_config(
            run_dir=run_dir,
            saved_model_path=saved_model,
            model_path=model,
            out_path=out,
            quant=quant,
            rep_coco=rep_coco,
            rep_images_dir=rep_images_dir,
            rep_max_images=rep_max_images,
            require_builtins_only=require_builtins_only,
        )
        artifacts = run_export_tflite(cfg)
    except (TFLiteConfigError, MissingTFLiteDependenciesError, TFLiteExportError) as exc:
        print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    print(f"[green]OK[/green] wrote {artifacts.tflite_path}")
    print(f"meta_json: {artifacts.metadata_path}")
    print(f"quant: {artifacts.quant}")
    if artifacts.builtin_ops_only is not None:
        print(f"builtin_ops_only: {str(artifacts.builtin_ops_only).lower()}")


@bench_app.command("tflite")
def bench_tflite_cli(
    run_dir: Annotated[
        Path | None,
        typer.Option(
            "--run-dir",
            exists=True,
            readable=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
    model: Annotated[Path | None, typer.Option("--model", exists=True, readable=True)] = None,
    images_dir: Annotated[
        Path | None,
        typer.Option(
            "--images-dir",
            exists=True,
            readable=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
    limit_images: Annotated[int, typer.Option("--limit-images")] = 16,
    warmup_runs: Annotated[int, typer.Option("--warmup-runs")] = 3,
    runs: Annotated[int, typer.Option("--runs")] = 16,
    out: Annotated[Path | None, typer.Option("--out")] = None,
):
    try:
        cfg = build_bench_tflite_config(
            run_dir=run_dir,
            model_path=model,
            out_path=out,
            images_dir=images_dir,
            limit_images=limit_images,
            warmup_runs=warmup_runs,
            runs=runs,
        )
        artifacts = run_bench_tflite(cfg)
    except (TFLiteConfigError, MissingTFLiteDependenciesError, TFLiteExportError) as exc:
        print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    print(f"[green]OK[/green] model={artifacts.model_path}")
    print(f"report_json: {artifacts.report_path}")


@inspect_app.command("tflite")
def inspect_tflite_cli(
    model: Annotated[Path, typer.Option("--model", exists=True, readable=True)],
):
    try:
        cfg = build_inspect_tflite_config(model_path=model)
        artifacts = run_inspect_tflite(cfg)
    except (TFLiteConfigError, MissingTFLiteDependenciesError, TFLiteExportError) as exc:
        print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    print(f"[green]OK[/green] model={artifacts.model_path}")
    print(f"builtin_ops_only: {str(artifacts.builtin_ops_only).lower()}")
    print(
        f"operator_names: {', '.join(artifacts.operator_names) if artifacts.operator_names else ''}"
    )
    print("inputs:")
    for item in artifacts.inputs:
        print(f"- {item['name']} shape={item['shape']} dtype={item['dtype']}")
    print("outputs:")
    for item in artifacts.outputs:
        print(f"- {item['name']} shape={item['shape']} dtype={item['dtype']}")


@golden_app.command("detect")
def golden_detect_cli(
    model: Annotated[Path, typer.Option("--model", exists=True, readable=True)],
    image: Annotated[Path, typer.Option("--image", exists=True, readable=True)],
    out: Annotated[Path, typer.Option("--out")],
    score_threshold: Annotated[float, typer.Option("--score-threshold")] = 0.3,
    max_results: Annotated[int, typer.Option("--max-results")] = 20,
):
    delegate_args = [
        "golden",
        "detect",
        "--model",
        str(model),
        "--image",
        str(image),
        "--out",
        str(out),
        "--score-threshold",
        str(score_threshold),
        "--max-results",
        str(max_results),
    ]
    delegated = _delegate_to_modelmaker_python(delegate_args)
    if delegated is not None:
        raise typer.Exit(code=delegated)

    try:
        cfg = build_golden_detect_config(
            model_path=model,
            image_path=image,
            out_path=out,
            score_threshold=score_threshold,
            max_results=max_results,
        )
        artifacts = generate_golden_detect(cfg)
    except (GoldenDetectConfigError, MissingGoldenDependenciesError) as exc:
        print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    print(f"[green]OK[/green] wrote {artifacts.out_path}")
    print(f"detections: {artifacts.num_detections}")


if __name__ == "__main__":
    app()
