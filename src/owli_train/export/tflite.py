from __future__ import annotations

from pathlib import Path

from owli_train.export.tflite_export import export_saved_model_to_tflite as _export


def export_saved_model_to_tflite(saved_model_dir: str | Path, out_path: str | Path) -> None:
    _export(saved_model_dir=saved_model_dir, out_path=out_path)
