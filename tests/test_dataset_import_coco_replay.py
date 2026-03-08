import json
from pathlib import Path

from typer.testing import CliRunner

from owli_train.cli import app
from owli_train.data.coco_replay import import_coco_replay_with_config

runner = CliRunner()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_replay_fixture(tmp_path: Path) -> Path:
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
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [0, 0, 20, 20]},
            {"id": 3, "image_id": 2, "category_id": 2, "bbox": [0, 0, 20, 20]},
            {"id": 4, "image_id": 2, "category_id": 3, "bbox": [0, 0, 5, 5]},
            {"id": 5, "image_id": 3, "category_id": 3, "bbox": [0, 0, 20, 20]},
            {"id": 6, "image_id": 4, "category_id": 4, "bbox": [0, 0, 20, 20]},
        ],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "car"},
            {"id": 3, "name": "bus"},
            {"id": 4, "name": "dog"},
        ],
    }
    coco_path = tmp_path / "instances_train2017.clean.json"
    _write_json(coco_path, coco)

    label_map = tmp_path / "coco_replay_to_ba.yaml"
    label_map.write_text(
        "\n".join(
            [
                "allowed_target_classes:",
                "  - bicycle",
                "  - bus",
                "  - car",
                "  - motorcycle",
                "  - person",
                "  - truck",
                "drop_unmapped: true",
                "map:",
                "  bicycle: bicycle",
                "  bus: bus",
                "  car: car",
                "  motorcycle: motorcycle",
                "  person: person",
                "  truck: truck",
            ]
        ),
        encoding="utf-8",
    )

    config = tmp_path / "coco_replay.yaml"
    config.write_text(
        "\n".join(
            [
                f"source_coco: {coco_path}",
                f"source_images_dir: {images_dir}",
                f"label_map: {label_map}",
                f"out_dir: {tmp_path / 'out'}",
                "selection:",
                "  min_bbox_min_side: 10",
                "  max_positive_images_per_class: 1",
            ]
        ),
        encoding="utf-8",
    )
    return config


def test_import_coco_replay_with_config_filters_to_rehearsal_classes_and_caps_images(
    tmp_path: Path,
) -> None:
    config_path = _make_replay_fixture(tmp_path)

    artifacts = import_coco_replay_with_config(config_path)

    replay = json.loads(artifacts.coco_path.read_text(encoding="utf-8"))
    qc_report = json.loads(artifacts.qc_report_path.read_text(encoding="utf-8"))

    assert [image["id"] for image in replay["images"]] == [1, 3]
    assert [category["name"] for category in replay["categories"]] == ["bus", "car", "person"]
    assert len(replay["annotations"]) == 3
    assert qc_report["filtered_unmapped_annotations"] == 1
    assert qc_report["filtered_small_bbox_annotations"] == 1
    assert qc_report["selected_image_counts"] == {"bus": 1, "car": 1, "person": 1}


def test_dataset_import_coco_replay_cli_writes_artifacts(tmp_path: Path) -> None:
    config_path = _make_replay_fixture(tmp_path)

    result = runner.invoke(app, ["dataset", "import", "coco-replay", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "wrote COCO replay" in result.stdout
    assert (tmp_path / "out" / "instances_ba_v1.coco.json").is_file()
    assert (tmp_path / "out" / "class_names.json").is_file()
    assert (tmp_path / "out" / "qc_report.json").is_file()
