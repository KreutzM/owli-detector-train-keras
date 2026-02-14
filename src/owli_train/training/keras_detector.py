from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    input_size: int
    num_classes: int
    class_names: list[str]
    coco_json: Path
    images_dir: Path
    seed: int
    batch_size: int
    epochs: int
    lr: float
    work_dir: Path


def train_detector(cfg: TrainConfig) -> None:
    raise NotImplementedError(
        'Training is not implemented in the base project yet. '
        'Use Codex to implement a Keras/KerasCV detector training pipeline.'
    )
