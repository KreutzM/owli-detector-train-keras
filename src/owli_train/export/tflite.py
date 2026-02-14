from __future__ import annotations

from pathlib import Path


def export_saved_model_to_tflite(saved_model_dir: str | Path, out_path: str | Path) -> None:
    raise NotImplementedError(
        "TFLite export is not implemented in the base project yet. "
        "Use Codex to implement SavedModel->TFLite export + quantization."
    )
