from __future__ import annotations

from pathlib import Path


def build_sample_repo(root: Path) -> Path:
    (root / "docs" / "reviews").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "label_contracts").mkdir(parents=True, exist_ok=True)
    (root / "work" / "datasets" / "demo-dataset").mkdir(parents=True, exist_ok=True)
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
    coco: ../work/datasets/demo-dataset/instances_materialized.json
    images_dir: ../work/datasets/demo-dataset/images
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
        "{}\n",
        encoding="utf-8",
    )
    (root / "work" / "datasets" / "demo-dataset" / "modelmaker.csv").write_text(
        "TRAIN,image.jpg,person,0,0,,,1,1\n",
        encoding="utf-8",
    )
    (root / "work" / "runs" / "20260309-123000-demo" / "artifacts" / "model.tflite").write_text(
        "demo\n",
        encoding="utf-8",
    )
    (root / "work" / "runs" / "20260309-123000-demo" / "reports" / "eval.md").write_text(
        "# Eval\n",
        encoding="utf-8",
    )

    return root
