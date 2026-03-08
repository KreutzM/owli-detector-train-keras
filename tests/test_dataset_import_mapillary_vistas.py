import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from owli_train.cli import app
from owli_train.data.mapillary_vistas import import_mapillary_vistas_to_coco

runner = CliRunner()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_mapillary_stub(root: Path) -> None:
    for split in ("training", "validation"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "panoptic").mkdir(parents=True)

    Image.new("RGB", (4000, 2000), color=(255, 0, 0)).save(
        root / "training" / "images" / "train_keep.jpg"
    )
    Image.new("RGB", (1000, 500), color=(0, 255, 0)).save(
        root / "training" / "images" / "train_drop.jpg"
    )
    Image.new("RGB", (800, 1600), color=(0, 0, 255)).save(
        root / "validation" / "images" / "val_keep.jpg"
    )

    categories = [
        {"id": 4, "name": "Fence", "supercategory": "construction--barrier--fence"},
        {"id": 20, "name": "Person", "supercategory": "human--person"},
        {"id": 21, "name": "Bicyclist", "supercategory": "human--rider--bicyclist"},
    ]

    train_payload = {
        "images": [
            {"id": "train_keep", "file_name": "train_keep.jpg", "width": 4000, "height": 2000},
            {"id": "train_drop", "file_name": "train_drop.jpg", "width": 1000, "height": 500},
        ],
        "annotations": [
            {
                "image_id": "train_keep",
                "file_name": "train_keep.png",
                "segments_info": [
                    {
                        "id": 1,
                        "category_id": 4,
                        "bbox": [100.0, 200.0, 500.0, 600.0],
                        "area": 300000,
                        "iscrowd": 0,
                    },
                    {
                        "id": 2,
                        "category_id": 21,
                        "bbox": [10.0, 10.0, 50.0, 80.0],
                        "area": 4000,
                        "iscrowd": 0,
                    },
                ],
            },
            {
                "image_id": "train_drop",
                "file_name": "train_drop.png",
                "segments_info": [
                    {
                        "id": 3,
                        "category_id": 21,
                        "bbox": [20.0, 20.0, 40.0, 40.0],
                        "area": 1600,
                        "iscrowd": 0,
                    },
                ],
            },
        ],
        "categories": categories,
    }
    val_payload = {
        "images": [
            {"id": "val_keep", "file_name": "val_keep.jpg", "width": 800, "height": 1600},
        ],
        "annotations": [
            {
                "image_id": "val_keep",
                "file_name": "val_keep.png",
                "segments_info": [
                    {
                        "id": 4,
                        "category_id": 20,
                        "bbox": [8.0, 16.0, 80.0, 160.0],
                        "area": 12800,
                        "iscrowd": 0,
                    },
                    {
                        "id": 5,
                        "category_id": 4,
                        "bbox": [50.0, 50.0, 0.0, 40.0],
                        "area": 0,
                        "iscrowd": 0,
                    },
                ],
            },
        ],
        "categories": categories,
    }

    _write_json(root / "training" / "panoptic" / "panoptic_2018.json", train_payload)
    _write_json(root / "validation" / "panoptic" / "panoptic_2018.json", val_payload)


def test_import_mapillary_vistas_to_coco_filters_and_scales(tmp_path: Path) -> None:
    source_root = tmp_path / "Map"
    _make_mapillary_stub(source_root)

    artifacts = import_mapillary_vistas_to_coco(
        mapillary_dir=source_root,
        out_dir=tmp_path / "out",
        label_map_path=Path("configs/label_maps/mapillary_vistas_to_ba.yaml"),
        max_long_side=1600,
    )

    combined = json.loads(artifacts.combined_coco_path.read_text(encoding="utf-8"))
    assert artifacts.train.images == 1
    assert artifacts.val.images == 1
    assert artifacts.train.annotations == 1
    assert artifacts.val.annotations == 1
    assert artifacts.train.skipped_images_without_labels == 1
    assert artifacts.val.skipped_annotations_invalid_bbox == 1

    file_names = sorted(item["file_name"] for item in combined["images"])
    assert file_names == ["training/train_keep.jpg", "validation/val_keep.jpg"]

    train_image = next(
        item for item in combined["images"] if item["file_name"] == "training/train_keep.jpg"
    )
    val_image = next(
        item for item in combined["images"] if item["file_name"] == "validation/val_keep.jpg"
    )
    assert train_image["width"] == 1600
    assert train_image["height"] == 800
    assert val_image["width"] == 800
    assert val_image["height"] == 1600

    train_ann = next(
        item for item in combined["annotations"] if item["image_id"] == train_image["id"]
    )
    val_ann = next(item for item in combined["annotations"] if item["image_id"] == val_image["id"])
    assert train_ann["bbox"] == [40.0, 80.0, 200.0, 240.0]
    assert val_ann["bbox"] == [8.0, 16.0, 80.0, 160.0]

    categories = [item["name"] for item in combined["categories"]]
    assert categories == [
        "obstacle_fence",
        "obstacle_hole",
        "obstacle_pole",
        "bicycle",
        "bus",
        "car",
        "motorcycle",
        "person",
        "truck",
    ]

    qc_report = json.loads(artifacts.qc_report_path.read_text(encoding="utf-8"))
    assert qc_report["splits"]["training"]["class_counts"] == {"obstacle_fence": 1}
    assert qc_report["splits"]["validation"]["class_counts"] == {"person": 1}


def test_dataset_import_mapillary_vistas_cli(tmp_path: Path) -> None:
    source_root = tmp_path / "Map"
    _make_mapillary_stub(source_root)
    out_dir = tmp_path / "processed"

    result = runner.invoke(
        app,
        [
            "dataset",
            "import",
            "mapillary-vistas",
            "--mapillary-dir",
            str(source_root),
            "--out-dir",
            str(out_dir),
            "--label-map",
            "configs/label_maps/mapillary_vistas_to_ba.yaml",
            "--max-long-side",
            "1600",
        ],
    )

    assert result.exit_code == 0
    assert (out_dir / "instances_ba_v1.coco.json").is_file()
    assert (out_dir / "annotations_train.coco.json").is_file()
    assert (out_dir / "annotations_val.coco.json").is_file()
    assert (out_dir / "splits.json").is_file()
    assert (out_dir / "qc_report.json").is_file()
    assert "train_images=1" in result.stdout
    assert "val_images=1" in result.stdout
