import json
from pathlib import Path

from typer.testing import CliRunner

from owli_train.cli import app
from owli_train.data.balance_coco import balance_coco_with_config

runner = CliRunner()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_split_json(path: Path, *, train: list[int], val: list[int], test: list[int]) -> None:
    payload = {"train": train, "val": val, "test": test}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_balancing_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True)
    for name in ("1.jpg", "2.jpg", "3.jpg", "4.jpg"):
        (images_dir / name).write_bytes(b"x")

    coco = {
        "images": [
            {"id": 1, "file_name": "1.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "2.jpg", "width": 100, "height": 100},
            {"id": 3, "file_name": "3.jpg", "width": 100, "height": 100},
            {"id": 4, "file_name": "4.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 2,
                "bbox": [0, 0, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [0, 0, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 2,
                "bbox": [0, 0, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 4,
                "image_id": 3,
                "category_id": 2,
                "bbox": [0, 0, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 5,
                "image_id": 4,
                "category_id": 1,
                "bbox": [0, 0, 5, 5],
                "area": 25,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "obstacle_hole"},
            {"id": 2, "name": "obstacle_pole"},
        ],
    }
    coco_path = tmp_path / "source.json"
    _write_json(coco_path, coco)

    splits_path = tmp_path / "splits.json"
    _write_split_json(splits_path, train=[1, 2], val=[3], test=[4])

    config_path = tmp_path / "balance.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"source_coco: {coco_path}",
                f"source_images_dir: {images_dir}",
                f"source_splits_json: {splits_path}",
                f"out_dir: {tmp_path / 'balanced'}",
                "selection:",
                "  min_bbox_min_side: 10",
                "  max_positive_images_per_class: 1",
            ]
        ),
        encoding="utf-8",
    )
    return config_path, coco_path, images_dir


def test_balance_coco_with_config_filters_small_boxes_and_caps_dominant_class(
    tmp_path: Path,
) -> None:
    config_path, _, _ = _make_balancing_fixture(tmp_path)

    artifacts = balance_coco_with_config(config_path)

    balanced = json.loads(artifacts.coco_path.read_text(encoding="utf-8"))
    qc_report = json.loads(artifacts.qc_report_path.read_text(encoding="utf-8"))
    splits = json.loads(artifacts.splits_path.read_text(encoding="utf-8"))

    assert [image["id"] for image in balanced["images"]] == [2]
    assert len(balanced["annotations"]) == 2
    assert [category["name"] for category in balanced["categories"]] == [
        "obstacle_hole",
        "obstacle_pole",
    ]
    assert qc_report["filtered_small_bbox_annotations"] == 1
    assert qc_report["class_selection_order"] == ["obstacle_hole", "obstacle_pole"]
    assert qc_report["selected_split_counts"] == {"train": 1}
    assert splits == {"train": [2], "val": [], "test": []}


def test_dataset_balance_coco_cli_writes_artifacts(tmp_path: Path) -> None:
    config_path, _, _ = _make_balancing_fixture(tmp_path)

    result = runner.invoke(app, ["dataset", "balance-coco", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "wrote balanced COCO" in result.stdout
    assert (tmp_path / "balanced" / "instances_ba_v1.coco.json").is_file()
    assert (tmp_path / "balanced" / "class_names.json").is_file()
    assert (tmp_path / "balanced" / "qc_report.json").is_file()
