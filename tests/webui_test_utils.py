from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def build_sample_repo(root: Path) -> Path:
    (root / "docs" / "reviews").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "label_contracts").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "data").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "smoke_coco" / "images").mkdir(parents=True, exist_ok=True)
    (root / "work" / "datasets" / "demo-dataset").mkdir(parents=True, exist_ok=True)
    (root / "work" / "datasets" / "demo-dataset" / "images").mkdir(parents=True, exist_ok=True)
    (root / "work" / "splits" / "demo-dataset").mkdir(parents=True, exist_ok=True)
    (root / "work" / "runs" / "20260309-123000-demo" / "artifacts").mkdir(
        parents=True, exist_ok=True
    )
    (root / "work" / "runs" / "20260309-123000-demo" / "reports").mkdir(parents=True, exist_ok=True)
    (root / "outputs").mkdir(parents=True, exist_ok=True)

    (root / "README.md").write_text("# Demo repo\n", encoding="utf-8")
    (root / "docs" / "runbook.md").write_text("# Runbook\n", encoding="utf-8")
    (root / "docs" / "MVP_Training_Plan.md").write_text("# Plan\n", encoding="utf-8")
    (root / "docs" / "BA_v1_Labelset.md").write_text("# BA v1\n", encoding="utf-8")
    (root / "docs" / "BA_v2_Hazard_Labelset.md").write_text("# BA v2\n", encoding="utf-8")
    (root / "docs" / "reviews" / "Codex-Task-Report_last.md").write_text(
        "# Report\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")

    image_path = root / "tests" / "smoke_coco" / "images" / "smoke1.jpg"
    Image.new("RGB", (10, 10), color=(220, 120, 80)).save(image_path)
    materialized_image_path = root / "work" / "datasets" / "demo-dataset" / "images" / "smoke1.jpg"
    Image.new("RGB", (10, 10), color=(220, 120, 80)).save(materialized_image_path)
    coco_payload = {
        "images": [
            {
                "id": 1,
                "file_name": "smoke1.jpg",
                "width": 10,
                "height": 10,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 1, 5, 5],
            }
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    (root / "tests" / "data" / "coco_min.json").write_text(
        json.dumps(coco_payload, indent=2),
        encoding="utf-8",
    )

    (root / "configs" / "label_contracts" / "ba_v1.yaml").write_text(
        """
version: ba_v1
purpose: Historical baseline.
class_names:
  - obstacle_bump
  - car
roles:
  ba_core:
    - obstacle_bump
  rehearsal:
    - car
classes:
  - name: obstacle_bump
    role: ba_core
    rationale: Baseline obstacle.
  - name: car
    role: rehearsal
    rationale: Baseline rehearsal.
out_of_scope:
  - Full COCO-80 coverage.
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "configs" / "label_contracts" / "ba_v2_hazard.yaml").write_text(
        """
version: ba_v2_hazard
status: preferred_product_target_pre_training_reset
purpose: Preferred hazard ontology.
class_names:
  - obstacle_ground
  - person
roles:
  hazard_core:
    - obstacle_ground
  rehearsal:
    - person
classes:
  - name: obstacle_ground
    role: hazard_core
    rationale: Ground hazard.
  - name: person
    role: rehearsal
    rationale: Rehearsal actor.
out_of_scope:
  - Generic scene understanding.
""".strip()
        + "\n",
        encoding="utf-8",
    )

    (root / "configs" / "balance_demo.yaml").write_text(
        """
source_coco: ../work/datasets/demo-dataset/instances_materialized.json
source_images_dir: ../work/datasets/demo-dataset/images
out_dir: ../work/datasets/demo-balanced
selection:
  min_bbox_min_side: 16
  max_positive_images_per_class: 100
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "configs" / "merge_demo.yaml").write_text(
        """
sources:
  - name: demo_source
    coco: ../tests/data/coco_min.json
    images_dir: ../tests/smoke_coco/images
    contract: label_contracts/ba_v2_hazard.yaml
settings:
  same_class_iou: 0.75
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "configs" / "efficientdet_demo.yaml").write_text(
        """
model:
  variant: lite2
data:
  csv: work/datasets/demo-dataset/modelmaker.csv
  images_dir: work/datasets/demo-dataset/images
  label_map_json: configs/label_contracts/ba_v2_hazard.class_names.json
outputs:
  work_dir: work
  out_dir: outputs
""".strip()
        + "\n",
        encoding="utf-8",
    )

    (root / "work" / "datasets" / "demo-dataset" / "instances_materialized.json").write_text(
        json.dumps(coco_payload, indent=2),
        encoding="utf-8",
    )
    (root / "work" / "datasets" / "demo-dataset" / "qc_report.json").write_text(
        json.dumps(
            {
                "summary": {
                    "images": 1,
                    "annotations": 1,
                    "categories": 1,
                    "small_bbox_filtered": 0,
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "work" / "datasets" / "demo-dataset" / "modelmaker.csv").write_text(
        "TRAIN,image.jpg,person,0,0,,,1,1\n",
        encoding="utf-8",
    )
    (root / "work" / "splits" / "demo-dataset" / "splits.json").write_text(
        json.dumps({"train": [1], "val": [], "test": []}, indent=2),
        encoding="utf-8",
    )
    (root / "work" / "runs" / "20260309-123000-demo" / "artifacts" / "model.tflite").write_text(
        "demo\n",
        encoding="utf-8",
    )
    (root / "work" / "runs" / "20260309-123000-demo" / "config_snapshot.yaml").write_text(
        "model:\n  variant: lite2\n",
        encoding="utf-8",
    )
    (root / "work" / "runs" / "20260309-123000-demo" / "mapping_snapshot.json").write_text(
        json.dumps({"person": 0}, indent=2),
        encoding="utf-8",
    )
    (root / "work" / "runs" / "20260309-123000-demo" / "reports" / "eval.md").write_text(
        "# Eval\n",
        encoding="utf-8",
    )
    (root / "work" / "runs" / "20260309-123000-demo" / "reports" / "eval_demo.json").write_text(
        json.dumps(
            {
                "created_at": "2026-03-09T12:30:00Z",
                "run_dir": "work/runs/20260309-123000-demo",
                "model_path": "work/runs/20260309-123000-demo/artifacts/model.tflite",
                "coco_path": "work/datasets/demo-dataset/instances_materialized.json",
                "images_dir": "tests/smoke_coco/images",
                "score_threshold": 0.1,
                "num_eval_images": 1,
                "num_detections": 1,
                "metrics": {"mAP50": 0.75, "precision": 1.0, "recall": 1.0},
                "summary_counts": {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0},
                "per_class": {
                    "person": {
                        "category_id": 1,
                        "tp": 1,
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "predictions": 1,
                        "ground_truth": 1,
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "work" / "runs" / "20260309-123000-demo" / "reports" / "eval_demo_alt.json").write_text(
        json.dumps(
            {
                "created_at": "2026-03-09T12:31:00Z",
                "metrics": {"mAP50": 0.62, "precision": 0.8, "recall": 0.9},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (
        root / "work" / "runs" / "20260309-123000-demo" / "reports" / "golden_obstacle4.json"
    ).write_text(
        json.dumps(
            {
                "created_at": "2026-03-09T12:32:00Z",
                "model_path": "work/runs/20260309-123000-demo/artifacts/model.tflite",
                "image_path": "tests/smoke_coco/images/smoke1.jpg",
                "contract": {
                    "class_labels_source": "configs/label_contracts/ba_v2_hazard.yaml",
                    "score_threshold": 0.1,
                    "max_results": 5,
                    "bbox_format": "xywh",
                    "coordinates": "absolute_pixels",
                },
                "model_metadata": {"name": "demo-model", "label_count": 1},
                "inspect_tflite": {"input_shape": [1, 320, 320, 3], "dtype": "uint8"},
                "detections": [
                    {
                        "class_index": 0,
                        "class_name": "person",
                        "score": 0.91,
                        "bbox": [1, 1, 5, 5],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return root
