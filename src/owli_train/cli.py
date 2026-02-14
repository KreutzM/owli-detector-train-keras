from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich import print

from owli_train.data.coco import load_coco, validate_coco
from owli_train.data.split import split_coco_image_ids, write_splits

app = typer.Typer(add_completion=False, no_args_is_help=True)
dataset_app = typer.Typer(no_args_is_help=True)
train_app = typer.Typer(no_args_is_help=True)
export_app = typer.Typer(no_args_is_help=True)

app.add_typer(dataset_app, name="dataset")
app.add_typer(train_app, name="train")
app.add_typer(export_app, name="export")


@dataset_app.command("validate")
def dataset_validate(coco: Path = typer.Option(..., "--coco", exists=True, readable=True)):
    obj = load_coco(coco)
    s = validate_coco(obj)
    print(f"[green]OK[/green] COCO: images={s.images}, ann={s.annotations}, cats={s.categories}")


@dataset_app.command("summarize")
def dataset_summarize(coco: Path = typer.Option(..., "--coco", exists=True, readable=True)):
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
    coco: Path = typer.Option(..., "--coco", exists=True, readable=True),
    out_dir: Path = typer.Option(Path("work/splits"), "--out-dir"),
    seed: int = typer.Option(1337, "--seed"),
    train_frac: float = typer.Option(0.8, "--train-frac"),
    val_frac: float = typer.Option(0.1, "--val-frac"),
):
    obj = load_coco(coco)
    _ = validate_coco(obj)
    splits = split_coco_image_ids(obj, seed=seed, train_frac=train_frac, val_frac=val_frac)
    out = write_splits(out_dir, splits)
    print(f"[green]OK[/green] wrote {out}")


@train_app.command("detect")
def train_detect(config: Path = typer.Option(..., "--config", exists=True, readable=True)):
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    print("[yellow]TODO[/yellow] training is not implemented in the base project yet.")
    print(cfg)


@export_app.command("tflite")
def export_tflite(
    saved_model: Path = typer.Option(..., "--saved-model"),
    out: Path = typer.Option(Path("outputs/model.tflite"), "--out"),
):
    print("[yellow]TODO[/yellow] export is not implemented in the base project yet.")
    print({"saved_model": str(saved_model), "out": str(out)})


if __name__ == "__main__":
    app()
