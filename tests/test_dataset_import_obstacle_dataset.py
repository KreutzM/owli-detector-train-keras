import json
from pathlib import Path

from PIL import Image

from owli_train.data.obstacle_dataset import import_obstacle_dataset_to_coco


def _write_xml(
    path: Path,
    *,
    filename: str,
    width: int,
    height: int,
    objects: list[tuple[str, tuple[int, int, int, int]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    object_xml = "\n".join(
        [
            (
                "  <object>\n"
                f"    <name>{name}</name>\n"
                "    <bndbox>\n"
                f"      <xmin>{bbox[0]}</xmin>\n"
                f"      <ymin>{bbox[1]}</ymin>\n"
                f"      <xmax>{bbox[2]}</xmax>\n"
                f"      <ymax>{bbox[3]}</ymax>\n"
                "    </bndbox>\n"
                "  </object>"
            )
            for name, bbox in objects
        ]
    )
    path.write_text(
        (
            "<annotation>\n"
            f"  <filename>{filename}</filename>\n"
            f"  <size><width>{width}</width><height>{height}</height><depth>3</depth></size>\n"
            f"{object_xml}\n"
            "</annotation>\n"
        ),
        encoding="utf-8",
    )


def _write_image(path: Path, *, width: int = 100, height: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(20, 40, 60)).save(path)


def _make_obstacle_dataset_stub(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / f"ann-{split}").mkdir(parents=True)
        (root / f"img-{split}").mkdir(parents=True)

    (root / "JPEGImages").mkdir(parents=True)

    _write_image(root / "img-train" / "train_local.jpg")
    _write_xml(
        root / "ann-train" / "train_local.xml",
        filename="train_local.jpg",
        width=100,
        height=80,
        objects=[("warning_column", (10, 12, 40, 70)), ("dog", (1, 1, 10, 10))],
    )

    _write_image(root / "JPEGImages" / "val_global.jpg")
    _write_xml(
        root / "ann-val" / "val_global.xml",
        filename="val_global.jpg",
        width=100,
        height=80,
        objects=[("motorbike", (15, 10, 50, 60)), ("car", (50, 15, 90, 70))],
    )

    _write_xml(
        root / "ann-test" / "test_missing.xml",
        filename="test_missing.jpg",
        width=100,
        height=80,
        objects=[("person", (10, 10, 30, 60))],
    )

    _write_image(root / "img-test" / "test_unmapped.jpg")
    _write_xml(
        root / "ann-test" / "test_unmapped.xml",
        filename="test_unmapped.jpg",
        width=100,
        height=80,
        objects=[("ashcan", (20, 20, 50, 70))],
    )


def test_import_obstacle_dataset_to_coco_uses_split_and_global_roots(tmp_path: Path) -> None:
    source_root = tmp_path / "Obstacle Dataset"
    _make_obstacle_dataset_stub(source_root)

    artifacts = import_obstacle_dataset_to_coco(
        dataset_dir=source_root,
        out_dir=tmp_path / "out",
        label_map_path=Path("configs/label_maps/obstacle_dataset_to_ba.yaml"),
        mode="copy",
    )

    combined = json.loads(artifacts.combined_coco_path.read_text(encoding="utf-8"))
    assert artifacts.categories == ["obstacle_pole", "car", "motorcycle"]
    assert artifacts.train.images == 1
    assert artifacts.train.annotations == 1
    assert artifacts.val.images == 1
    assert artifacts.val.annotations == 2
    assert artifacts.test.images == 0
    assert artifacts.test.skipped_missing_images == 1
    assert artifacts.test.skipped_images_without_mapped_annotations == 1

    file_names = sorted(item["file_name"] for item in combined["images"])
    assert file_names == ["train/train_local.jpg", "val/val_global.jpg"]

    image_paths = [
        tmp_path / "out" / "images" / "train" / "train_local.jpg",
        tmp_path / "out" / "images" / "val" / "val_global.jpg",
    ]
    assert all(path.is_file() for path in image_paths)

    category_names = {item["id"]: item["name"] for item in combined["categories"]}
    observed_categories = {category_names[ann["category_id"]] for ann in combined["annotations"]}
    assert observed_categories == {"obstacle_pole", "car", "motorcycle"}

    qc_report = json.loads(artifacts.qc_report_path.read_text(encoding="utf-8"))
    assert qc_report["splits"]["test"]["missing_image_samples"] == ["test_missing.jpg"]
    assert qc_report["splits"]["train"]["unmapped_source_class_counts"] == {"dog": 1}
