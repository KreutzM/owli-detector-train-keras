# Architecture Baseline

## Scope
- Dataset foundation for COCO: validate, normalize, split.
- Baseline detector training via TensorFlow + KerasCV (YOLOv8).

## Modules
- `src/owli_train/cli.py`: Typer command surface.
- `src/owli_train/data/coco.py`: COCO load/validate/normalize/write helpers.
- `src/owli_train/data/split.py`: deterministic split logic and split-file emission.
- `src/owli_train/training/keras_detector.py`: config parsing, COCO-to-tf.data, preprocessing, KerasCV model build, run management.

## Dataset flow
1. Load JSON from disk.
2. Validate structure and references:
   - required top-level keys (`images`, `annotations`, `categories`)
   - unique IDs for images/annotations/categories
   - annotation `image_id` and `category_id` references
   - bbox sanity (`width > 0`, `height > 0`)
   - optional image-file existence checks via `--images-dir`
3. Normalize categories (optional label-map merges), then write normalized JSON.
4. Generate deterministic train/val/test splits by image IDs and optionally materialize split COCO files.

## Determinism
- Split IDs are sorted before seeded shuffle.
- Normalized category IDs are rebuilt in sorted name order (contiguous IDs starting at 1).
- Training run seeds are applied to Python, NumPy, and TensorFlow.

## Testing constraints
- Tests are offline, small-fixture, and GPU-free.
- No TensorFlow/Keras imports are required for dataset tests.
- Default test suite validates training config + CLI wiring without importing TensorFlow.
