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
    (root / "configs" / "efficientdet_lite2_ba_mvp_stage3.yaml").write_text(
        """
model:
  variant: lite2
data:
  csv: work/datasets/ba_mvp_stage3_balanced_multisource/modelmaker.csv
  images_dir: work/datasets/ba_mvp_stage3_balanced_multisource/images
  label_map_json: configs/label_contracts/ba_v1.class_names.json
outputs:
  work_dir: work
  out_dir: outputs
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "configs" / "efficientdet_lite2_ba_mvp_stage4.yaml").write_text(
        """
model:
  variant: lite2
data:
  csv: work/datasets/ba_mvp_stage4_with_coco_replay/modelmaker.csv
  images_dir: work/datasets/ba_mvp_stage4_with_coco_replay/images
  label_map_json: configs/label_contracts/ba_v1.class_names.json
outputs:
  work_dir: work
  out_dir: outputs
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "configs" / "efficientdet_lite2_ba_mvp_stage3_plus_crops.yaml").write_text(
        """
model:
  variant: lite2
data:
  csv: work/datasets/ba_mvp_stage3_plus_crops/modelmaker.csv
  images_dir: work/datasets/ba_mvp_stage3_plus_crops/images
  label_map_json: configs/label_contracts/ba_v1.class_names.json
outputs:
  work_dir: work
  out_dir: outputs
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def _write_run(
        run_id: str,
        *,
        config_name: str,
        eval_reports: dict[str, dict[str, object]],
        golden_name: str | None = None,
    ) -> None:
        run_dir = root / "work" / "runs" / run_id
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (run_dir / "reports").mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts" / "model.tflite").write_text("demo\n", encoding="utf-8")
        config_text = (root / "configs" / config_name).read_text(encoding="utf-8")
        (run_dir / "config.yaml").write_text(config_text, encoding="utf-8")
        (run_dir / "mapping_files.json").write_text(
            json.dumps({"config_name": config_name}, indent=2),
            encoding="utf-8",
        )
        for report_name, payload in eval_reports.items():
            (run_dir / "reports" / report_name).write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        if golden_name is not None:
            (run_dir / "reports" / golden_name).write_text(
                json.dumps(
                    {
                        "created_at": "2026-03-09T12:32:00Z",
                        "model_path": f"work/runs/{run_id}/artifacts/model.tflite",
                        "image_path": "tests/smoke_coco/images/smoke1.jpg",
                        "contract": {
                            "class_labels_source": "configs/label_contracts/ba_v1.yaml",
                            "score_threshold": 0.1,
                            "max_results": 5,
                            "bbox_format": "xywh",
                            "coordinates": "absolute_pixels",
                        },
                        "model_metadata": {"name": run_id, "label_count": 3},
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

    (root / "work" / "splits" / "ba_mvp_stage3_balanced_multisource").mkdir(
        parents=True, exist_ok=True
    )
    (root / "work" / "splits" / "ba_mvp_stage4_with_coco_replay").mkdir(parents=True, exist_ok=True)
    (
        root / "work" / "splits" / "ba_mvp_stage3_balanced_multisource" / "instances_test.json"
    ).write_text(
        json.dumps(coco_payload, indent=2),
        encoding="utf-8",
    )
    (
        root / "work" / "splits" / "ba_mvp_stage4_with_coco_replay" / "instances_test.json"
    ).write_text(
        json.dumps(coco_payload, indent=2),
        encoding="utf-8",
    )

    _write_run(
        "20260308-183140-ba-mvp-stage3-20260308",
        config_name="efficientdet_lite2_ba_mvp_stage3.yaml",
        eval_reports={
            "eval_efficientdet_tflite_stage3_test.json": {
                "created_at": "2026-03-08T18:40:00Z",
                "run_dir": "work/runs/20260308-183140-ba-mvp-stage3-20260308",
                "model_path": "work/runs/20260308-183140-ba-mvp-stage3-20260308/artifacts/model.tflite",
                "coco_path": "work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json",
                "images_dir": "tests/smoke_coco/images",
                "score_threshold": 0.1,
                "num_eval_images": 408,
                "num_detections": 7059,
                "metrics": {
                    "AP": 0.1307,
                    "AP50": 0.2325,
                    "AP75": 0.127,
                    "AR100": 0.217,
                },
                "summary_counts": {
                    "tp": 1447,
                    "fp": 5612,
                    "fn": 2427,
                    "precision": 0.205,
                    "recall": 0.3735,
                },
                "per_class": {
                    "obstacle_bump": {
                        "tp": 22,
                        "fp": 524,
                        "fn": 20,
                        "precision": 0.0403,
                        "recall": 0.5238,
                    },
                    "obstacle_fence": {
                        "tp": 23,
                        "fp": 350,
                        "fn": 92,
                        "precision": 0.0617,
                        "recall": 0.2,
                    },
                    "obstacle_hole": {
                        "tp": 19,
                        "fp": 495,
                        "fn": 30,
                        "precision": 0.037,
                        "recall": 0.3878,
                    },
                    "obstacle_pole": {
                        "tp": 163,
                        "fp": 962,
                        "fn": 964,
                        "precision": 0.1449,
                        "recall": 0.1446,
                    },
                    "person": {
                        "tp": 344,
                        "fp": 1149,
                        "fn": 351,
                        "precision": 0.2304,
                        "recall": 0.495,
                    },
                    "car": {
                        "tp": 553,
                        "fp": 1245,
                        "fn": 417,
                        "precision": 0.3076,
                        "recall": 0.5701,
                    },
                    "truck": {"tp": 49, "fp": 172, "fn": 93, "precision": 0.2217, "recall": 0.3451},
                },
            }
        },
        golden_name="golden_obstacle4.json",
    )
    _write_run(
        "20260308-211806-ba-mvp-stage4-20260308",
        config_name="efficientdet_lite2_ba_mvp_stage4.yaml",
        eval_reports={
            "eval_efficientdet_tflite_stage4_test.json": {
                "created_at": "2026-03-08T21:20:00Z",
                "run_dir": "work/runs/20260308-211806-ba-mvp-stage4-20260308",
                "model_path": "work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite",
                "coco_path": "work/splits/ba_mvp_stage4_with_coco_replay/instances_test.json",
                "images_dir": "tests/smoke_coco/images",
                "score_threshold": 0.1,
                "num_eval_images": 420,
                "num_detections": 6500,
                "metrics": {
                    "AP": 0.14,
                    "AP50": 0.24,
                    "AP75": 0.13,
                    "AR100": 0.22,
                },
                "summary_counts": {
                    "tp": 1500,
                    "fp": 5000,
                    "fn": 2300,
                    "precision": 0.23,
                    "recall": 0.395,
                },
                "per_class": {
                    "obstacle_bump": {
                        "tp": 21,
                        "fp": 500,
                        "fn": 21,
                        "precision": 0.0403,
                        "recall": 0.5,
                    },
                    "obstacle_fence_rail": {
                        "tp": 25,
                        "fp": 332,
                        "fn": 90,
                        "precision": 0.07,
                        "recall": 0.22,
                    },
                    "obstacle_hole_dropoff": {
                        "tp": 18,
                        "fp": 360,
                        "fn": 31,
                        "precision": 0.0476,
                        "recall": 0.3673,
                    },
                    "obstacle_pole": {
                        "tp": 170,
                        "fp": 1040,
                        "fn": 957,
                        "precision": 0.1405,
                        "recall": 0.1508,
                    },
                    "person": {"tp": 350, "fp": 1110, "fn": 345, "precision": 0.24, "recall": 0.51},
                    "car": {"tp": 565, "fp": 1258, "fn": 405, "precision": 0.31, "recall": 0.59},
                    "bus": {"tp": 69, "fp": 160, "fn": 101, "precision": 0.3013, "recall": 0.4059},
                },
            },
            "eval_efficientdet_tflite_stage3_test.json": {
                "created_at": "2026-03-08T21:22:00Z",
                "run_dir": "work/runs/20260308-211806-ba-mvp-stage4-20260308",
                "model_path": "work/runs/20260308-211806-ba-mvp-stage4-20260308/artifacts/model.tflite",
                "coco_path": "work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json",
                "images_dir": "tests/smoke_coco/images",
                "score_threshold": 0.1,
                "num_eval_images": 408,
                "num_detections": 6900,
                "metrics": {
                    "AP": 0.129,
                    "AP50": 0.229,
                    "AP75": 0.121,
                    "AR100": 0.214,
                },
                "summary_counts": {
                    "tp": 1434,
                    "fp": 5466,
                    "fn": 2440,
                    "precision": 0.2078,
                    "recall": 0.3702,
                },
                "per_class": {
                    "obstacle_bump": {
                        "tp": 20,
                        "fp": 508,
                        "fn": 22,
                        "precision": 0.0379,
                        "recall": 0.4762,
                    },
                    "obstacle_fence_rail": {
                        "tp": 27,
                        "fp": 319,
                        "fn": 88,
                        "precision": 0.078,
                        "recall": 0.24,
                    },
                    "obstacle_hole_dropoff": {
                        "tp": 17,
                        "fp": 349,
                        "fn": 32,
                        "precision": 0.0464,
                        "recall": 0.3469,
                    },
                    "obstacle_pole": {
                        "tp": 168,
                        "fp": 1001,
                        "fn": 959,
                        "precision": 0.1437,
                        "recall": 0.1491,
                    },
                    "person": {
                        "tp": 337,
                        "fp": 1084,
                        "fn": 358,
                        "precision": 0.237,
                        "recall": 0.504,
                    },
                    "car": {"tp": 546, "fp": 1228, "fn": 424, "precision": 0.308, "recall": 0.563},
                    "bus": {"tp": 67, "fp": 167, "fn": 103, "precision": 0.2863, "recall": 0.3941},
                },
            },
        },
        golden_name="golden_obstacle4.json",
    )
    _write_run(
        "20260309-072510-ba-mvp-stage3-plus-crops-20260309",
        config_name="efficientdet_lite2_ba_mvp_stage3_plus_crops.yaml",
        eval_reports={
            "eval_efficientdet_tflite_stage3_test.json": {
                "created_at": "2026-03-09T07:35:00Z",
                "run_dir": "work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309",
                "model_path": "work/runs/20260309-072510-ba-mvp-stage3-plus-crops-20260309/artifacts/model.tflite",
                "coco_path": "work/splits/ba_mvp_stage3_balanced_multisource/instances_test.json",
                "images_dir": "tests/smoke_coco/images",
                "score_threshold": 0.1,
                "num_eval_images": 408,
                "num_detections": 6851,
                "metrics": {
                    "AP": 0.128,
                    "AP50": 0.2276,
                    "AP75": 0.1202,
                    "AR100": 0.2142,
                },
                "summary_counts": {
                    "tp": 1427,
                    "fp": 5424,
                    "fn": 2447,
                    "precision": 0.2083,
                    "recall": 0.3684,
                },
                "per_class": {
                    "obstacle_bump": {
                        "tp": 16,
                        "fp": 415,
                        "fn": 26,
                        "precision": 0.0371,
                        "recall": 0.381,
                    },
                    "obstacle_fence": {
                        "tp": 29,
                        "fp": 308,
                        "fn": 86,
                        "precision": 0.0861,
                        "recall": 0.2522,
                    },
                    "obstacle_hole_dropoff": {
                        "tp": 17,
                        "fp": 356,
                        "fn": 32,
                        "precision": 0.0456,
                        "recall": 0.3469,
                    },
                    "obstacle_pole": {
                        "tp": 172,
                        "fp": 1177,
                        "fn": 955,
                        "precision": 0.1275,
                        "recall": 0.1526,
                    },
                    "person": {
                        "tp": 330,
                        "fp": 1066,
                        "fn": 365,
                        "precision": 0.2364,
                        "recall": 0.4748,
                    },
                    "car": {
                        "tp": 539,
                        "fp": 1210,
                        "fn": 431,
                        "precision": 0.3082,
                        "recall": 0.5557,
                    },
                    "bicycle": {
                        "tp": 79,
                        "fp": 226,
                        "fn": 163,
                        "precision": 0.259,
                        "recall": 0.3264,
                    },
                },
            }
        },
        golden_name="golden_obstacle4.json",
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
