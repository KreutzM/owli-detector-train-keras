import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from owli_train.cli import app
from owli_train.data.coco_crops import export_coco_crops_with_config

runner = CliRunner()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_image(path: Path, *, size: tuple[int, int] = (100, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 140, 160)).save(path)


def _write_crop_config(
    path: Path,
    *,
    source_coco: Path,
    source_images_dir: Path,
    out_dir: Path,
    target_classes: list[str],
    allowed_source_prefixes: list[str],
    max_bbox_min_side: int = 20,
    max_bbox_area_ratio: float = 0.05,
    max_bbox_short_side_ratio: float = 0.2,
    max_crops_per_class: int = 5,
    max_crops_per_image: int = 1,
    context_scale: float = 4.0,
    min_size: int = 40,
    max_size: int = 40,
    min_retained_area_ratio: float = 0.5,
    min_retained_bbox_min_side: int = 4,
    file_name_prefix: str = "stage3_crops",
) -> None:
    lines = [
        f"source_coco: {source_coco}",
        f"source_images_dir: {source_images_dir}",
        f"out_dir: {out_dir}",
        "selection:",
        "  target_classes:",
        *[f"    - {name}" for name in target_classes],
        "  allowed_source_prefixes:",
        *[f"    - {name}" for name in allowed_source_prefixes],
        f"  max_bbox_min_side: {max_bbox_min_side}",
        f"  max_bbox_area_ratio: {max_bbox_area_ratio}",
        f"  max_bbox_short_side_ratio: {max_bbox_short_side_ratio}",
        f"  max_crops_per_class: {max_crops_per_class}",
        f"  max_crops_per_image: {max_crops_per_image}",
        "crop:",
        f"  context_scale: {context_scale}",
        f"  min_size: {min_size}",
        f"  max_size: {max_size}",
        f"  min_retained_area_ratio: {min_retained_area_ratio}",
        f"  min_retained_bbox_min_side: {min_retained_bbox_min_side}",
        f"  file_name_prefix: {file_name_prefix}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_export_coco_crops_with_config_reprojects_boxes_and_filters_partial_boxes(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "mapillary_vistas" / "img1.jpg")

    coco = {
        "images": [
            {"id": 1, "file_name": "mapillary_vistas/img1.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [40, 40, 10, 10], "area": 100},
            {"id": 2, "image_id": 1, "category_id": 5, "bbox": [45, 45, 20, 20], "area": 400},
            {"id": 3, "image_id": 1, "category_id": 3, "bbox": [58, 58, 18, 18], "area": 324},
        ],
        "categories": [
            {"id": 2, "name": "obstacle_pole"},
            {"id": 3, "name": "obstacle_hole"},
            {"id": 5, "name": "person"},
        ],
    }
    coco_path = tmp_path / "instances_train.json"
    _write_json(coco_path, coco)

    config_path = tmp_path / "crops.yaml"
    _write_crop_config(
        config_path,
        source_coco=coco_path,
        source_images_dir=images_dir,
        out_dir=tmp_path / "out",
        target_classes=["obstacle_pole"],
        allowed_source_prefixes=["mapillary_vistas"],
    )

    artifacts = export_coco_crops_with_config(config_path)

    exported = json.loads(artifacts.coco_path.read_text(encoding="utf-8"))
    assert len(exported["images"]) == 1
    assert len(exported["annotations"]) == 2
    assert [category["name"] for category in exported["categories"]] == ["obstacle_pole", "person"]

    image = exported["images"][0]
    assert (
        image["file_name"] == "stage3_crops/obstacle_pole/mapillary_vistas/0000001_ann0000001.jpg"
    )
    assert image["crop_box"] == [25, 25, 40, 40]

    ann_by_name = {
        category["name"]: annotation
        for annotation in exported["annotations"]
        for category in exported["categories"]
        if int(category["id"]) == int(annotation["category_id"])
    }
    assert ann_by_name["obstacle_pole"]["bbox"] == [15.0, 15.0, 10.0, 10.0]
    assert ann_by_name["person"]["bbox"] == [20.0, 20.0, 20.0, 20.0]

    qc_report = json.loads(artifacts.qc_report_path.read_text(encoding="utf-8"))
    assert qc_report["selected_crops_by_class"] == {"obstacle_pole": 1}
    assert qc_report["selected_crops_by_source"] == {"mapillary_vistas": 1}


def test_export_coco_crops_with_config_applies_caps_and_keeps_category_ids_stable(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "images"
    for name in ("img1.jpg", "img2.jpg", "img3.jpg"):
        _write_image(images_dir / "obstacle4" / name)

    coco = {
        "images": [
            {"id": 1, "file_name": "obstacle4/img1.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "obstacle4/img2.jpg", "width": 100, "height": 100},
            {"id": 3, "file_name": "obstacle4/img3.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 4, "bbox": [40, 40, 10, 10], "area": 100},
            {"id": 2, "image_id": 2, "category_id": 4, "bbox": [10, 10, 10, 10], "area": 100},
            {"id": 3, "image_id": 3, "category_id": 4, "bbox": [10, 10, 40, 40], "area": 1600},
            {"id": 4, "image_id": 2, "category_id": 9, "bbox": [12, 12, 12, 12], "area": 144},
            {"id": 5, "image_id": 1, "category_id": 4, "bbox": [40, 40, 10, 10], "area": 100},
        ],
        "categories": [
            {"id": 4, "name": "obstacle_pole"},
            {"id": 9, "name": "person"},
        ],
    }
    coco_path = tmp_path / "instances_train.json"
    _write_json(coco_path, coco)

    config_path = tmp_path / "crops.yaml"
    _write_crop_config(
        config_path,
        source_coco=coco_path,
        source_images_dir=images_dir,
        out_dir=tmp_path / "out",
        target_classes=["obstacle_pole"],
        allowed_source_prefixes=["obstacle4"],
        max_crops_per_class=1,
        max_bbox_area_ratio=0.02,
        max_bbox_short_side_ratio=0.15,
    )

    artifacts = export_coco_crops_with_config(config_path)

    exported = json.loads(artifacts.coco_path.read_text(encoding="utf-8"))
    class_names = json.loads(artifacts.class_names_path.read_text(encoding="utf-8"))
    qc_report = json.loads(artifacts.qc_report_path.read_text(encoding="utf-8"))

    assert len(exported["images"]) == 1
    assert qc_report["candidate_annotations_by_class"] == {"obstacle_pole": 3}
    assert qc_report["selected_crops_by_class"] == {"obstacle_pole": 1}
    assert qc_report["dropped_duplicate_same_class_annotations"] == 1
    assert [category["id"] for category in exported["categories"]] == [4]
    assert class_names == {"class_names": ["obstacle_pole"], "category_ids": [4]}


def test_dataset_export_coco_crops_cli_writes_artifacts(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "od_ba_v1" / "img1.jpg")

    coco = {
        "images": [
            {"id": 1, "file_name": "od_ba_v1/img1.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [20, 20, 12, 12], "area": 144},
        ],
        "categories": [
            {"id": 3, "name": "obstacle_hole"},
        ],
    }
    coco_path = tmp_path / "instances_train.json"
    _write_json(coco_path, coco)

    config_path = tmp_path / "crops.yaml"
    _write_crop_config(
        config_path,
        source_coco=coco_path,
        source_images_dir=images_dir,
        out_dir=tmp_path / "out",
        target_classes=["obstacle_hole"],
        allowed_source_prefixes=["od_ba_v1"],
    )

    result = runner.invoke(app, ["dataset", "export", "coco-crops", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "wrote crop COCO" in result.stdout
    assert (tmp_path / "out" / "instances_ba_v1.coco.json").is_file()
    assert (tmp_path / "out" / "class_names.json").is_file()
    assert (tmp_path / "out" / "qc_report.json").is_file()
