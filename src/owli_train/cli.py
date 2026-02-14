from __future__ import annotations

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
from owli_train.data.split import split_coco_image_ids, write_split_coco_files, write_splits
from owli_train.eval.detect import (
    EvalConfigError,
    MissingEvalDependenciesError,
    build_eval_detect_config,
    evaluate_detect,
)
from owli_train.export.tflite_export import (
    MissingTFLiteDependenciesError,
    TFLiteConfigError,
    TFLiteExportError,
    build_bench_tflite_config,
    build_export_tflite_config,
)
from owli_train.export.tflite_export import (
    bench_tflite as run_bench_tflite,
)
from owli_train.export.tflite_export import (
    export_tflite as run_export_tflite,
)
from owli_train.training.keras_detector import (
    MissingTrainingDependenciesError,
    TrainingError,
    train_detector_from_config,
)

DEFAULT_NORMALIZED_OUT = Path("work/normalized/instances.json")
DEFAULT_SPLITS_OUT_DIR = Path("work/splits")

app = typer.Typer(add_completion=False, no_args_is_help=True)
dataset_app = typer.Typer(no_args_is_help=True)
train_app = typer.Typer(no_args_is_help=True)
eval_app = typer.Typer(no_args_is_help=True)
export_app = typer.Typer(no_args_is_help=True)
bench_app = typer.Typer(no_args_is_help=True)

app.add_typer(dataset_app, name="dataset")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(export_app, name="export")
app.add_typer(bench_app, name="bench")


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


@train_app.command("detect")
def train_detect(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True)],
    run_name: Annotated[str | None, typer.Option("--run-name")] = None,
    max_steps: Annotated[int | None, typer.Option("--max-steps")] = None,
    limit_train_images: Annotated[int | None, typer.Option("--limit-train-images")] = None,
    limit_val_images: Annotated[int | None, typer.Option("--limit-val-images")] = None,
    resume: Annotated[Path | None, typer.Option("--resume", exists=True, readable=True)] = None,
):
    try:
        artifacts = train_detector_from_config(
            config_path=config,
            run_name=run_name,
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
        )
        artifacts = run_export_tflite(cfg)
    except (TFLiteConfigError, MissingTFLiteDependenciesError, TFLiteExportError) as exc:
        print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    print(f"[green]OK[/green] wrote {artifacts.tflite_path}")
    print(f"meta_json: {artifacts.metadata_path}")
    print(f"quant: {artifacts.quant}")


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


if __name__ == "__main__":
    app()
