import json
from pathlib import Path

import yaml
from PIL import Image

from owli_train.data.coco_replay import import_coco_replay_with_config
from owli_train.data.mapillary_vistas import import_mapillary_vistas_to_coco
from owli_train.data.obstacle_dataset import import_obstacle_dataset_to_coco


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_stub_images(root: Path) -> None:
    Image.new("RGB", (4000, 2000), color=(255, 0, 0)).save(
        root / "training" / "images" / "train_keep.jpg"
    )
    Image.new("RGB", (1000, 500), color=(0, 255, 0)).save(
        root / "training" / "images" / "train_drop.jpg"
    )
    Image.new("RGB", (800, 1600), color=(0, 0, 255)).save(
        root / "validation" / "images" / "val_keep.jpg"
    )


def _make_mapillary_stub(root: Path) -> None:
    for split in ("training", "validation"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "panoptic").mkdir(parents=True)

    _write_stub_images(root)

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
                    }
                ],
            }
        ],
        "categories": categories,
    }

    _write_json(root / "training" / "panoptic" / "panoptic_2018.json", train_payload)
    _write_json(root / "validation" / "panoptic" / "panoptic_2018.json", val_payload)


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

    config = tmp_path / "coco_replay.yaml"
    config.write_text(
        "\n".join(
            [
                f"source_coco: {coco_path}",
                f"source_images_dir: {images_dir}",
                f"label_map: {Path('configs/label_maps/coco_replay_to_ba_v2_hazard.yaml').resolve()}",
                f"out_dir: {tmp_path / 'out'}",
                "selection:",
                "  min_bbox_min_side: 10",
                "  max_positive_images_per_class: 1",
            ]
        ),
        encoding="utf-8",
    )
    return config


def test_ba_v2_mapillary_slice_stays_inside_contract_and_keeps_open_gaps_open() -> None:
    contract = _load_yaml(Path("configs/label_contracts/ba_v2_hazard.yaml"))
    prep = _load_yaml(Path("configs/label_maps/mapillary_vistas_to_ba_v2_hazard.yaml"))

    assert prep["status"] == "first_ba_v2_hazard_slice_from_reviewed_vistas_taxonomy"
    assert prep["drop_unmapped"] is True
    assert set(prep["allowed_target_classes"]) <= set(contract["class_names"])
    assert set(prep["map"].values()) <= set(contract["class_names"])
    assert prep["map"]["construction--barrier--fence"] == "obstacle_barrier"
    assert prep["map"]["object--pothole"] == "obstacle_hole_dropoff"
    assert "object--manhole" not in prep["map"]
    assert "human--rider--bicyclist" not in prep["map"]


def test_ba_v2_od_slice_stays_inside_contract_and_remains_pole_first() -> None:
    contract = _load_yaml(Path("configs/label_contracts/ba_v2_hazard.yaml"))
    prep = _load_yaml(Path("configs/label_maps/obstacle_dataset_to_ba_v2_hazard.yaml"))

    assert prep["status"] == "first_ba_v2_hazard_slice_from_reviewed_od_taxonomy"
    assert prep["drop_unmapped"] is True
    assert set(prep["allowed_target_classes"]) <= set(contract["class_names"])
    assert set(prep["map"].values()) <= set(contract["class_names"])
    assert prep["map"]["pole"] == "obstacle_pole"
    assert prep["map"]["warning_column"] == "obstacle_pole"
    assert prep["map"]["motorbike"] == "motorcycle"
    assert "ashcan" not in prep["map"]
    assert "fire_hydrant" not in prep["map"]


def test_ba_v2_coco_replay_slice_stays_rehearsal_only() -> None:
    contract = _load_yaml(Path("configs/label_contracts/ba_v2_hazard.yaml"))
    prep = _load_yaml(Path("configs/label_maps/coco_replay_to_ba_v2_hazard.yaml"))

    assert prep["status"] == "first_ba_v2_hazard_rehearsal_slice"
    assert prep["drop_unmapped"] is True
    assert prep["allowed_target_classes"] == contract["roles"]["rehearsal"]
    assert prep["map"] == {name: name for name in contract["roles"]["rehearsal"]}


def test_ba_v1_non_obstacle4_export_remap_stays_narrow_and_skips_obstacle_bump() -> None:
    prep = _load_yaml(Path("configs/label_maps/ba_v1_non_obstacle4_export_to_ba_v2_hazard.yaml"))

    assert prep["map"]["obstacle_fence"] == "obstacle_barrier"
    assert prep["map"]["obstacle_hole"] == "obstacle_hole_dropoff"
    assert prep["map"]["obstacle_pole"] == "obstacle_pole"
    assert "obstacle_bump" not in prep["map"]


def test_obstacle4_ground_bootstrap_map_stays_narrow_and_mvp_aligned() -> None:
    contract = _load_yaml(Path("configs/label_contracts/ba_v2_hazard.yaml"))
    prep = _load_yaml(Path("configs/label_maps/obstacle4_gt_to_ba_v2_hazard_ground_bootstrap.yaml"))

    assert prep["status"] == "narrow_local_obstacle4_ground_bootstrap"
    assert prep["drop_unmapped"] is True
    assert set(prep["allowed_target_classes"]) <= set(contract["class_names"])
    assert set(prep["map"].values()) <= set(contract["class_names"])
    assert prep["map"] == {
        "obstacle_bump": "obstacle_ground",
        "obstacle_fence": "obstacle_barrier",
        "obstacle_hole": "obstacle_hole_dropoff",
        "obstacle_pole": "obstacle_pole",
    }
    assert prep["allowed_target_classes"] == contract["roles"]["hazard_core"]


def test_ba_v2_slice01_configs_point_to_mapillary_and_od_only() -> None:
    balance_map = _load_yaml(Path("configs/balance_ba_v2_hazard_mapillary_slice01.yaml"))
    balance_od = _load_yaml(Path("configs/balance_ba_v2_hazard_od_slice01.yaml"))
    manifest = _load_yaml(Path("configs/merge_ba_v2_hazard_slice01_mapillary_od.yaml"))

    assert (
        balance_map["source_coco"]
        == "../work/datasets/mapillary_vistas_ba_v2_hazard_source/instances_normalized.json"
    )
    assert balance_map["selection"]["max_positive_images_per_class"] == 300
    assert (
        balance_od["source_coco"]
        == "../work/datasets/od_ba_v2_hazard_source/instances_normalized.json"
    )
    assert balance_od["selection"]["max_positive_images_per_class"] == 100000
    assert [source["name"] for source in manifest["sources"]] == [
        "mapillary_vistas_ba_v2_hazard_slice01_balanced",
        "od_ba_v2_hazard",
    ]
    assert all(
        source["contract"] == "label_contracts/ba_v2_hazard.yaml" for source in manifest["sources"]
    )


def test_ba_v2_slice02_configs_add_ground_bootstrap_without_claiming_overhang() -> None:
    balance = _load_yaml(Path("configs/balance_ba_v2_hazard_obstacle4_ground_slice02.yaml"))
    merge = _load_yaml(
        Path("configs/merge_ba_v2_hazard_slice02_mapillary_od_obstacle4_ground.yaml")
    )
    materialize = _load_yaml(
        Path("configs/merge_ba_v2_hazard_slice02_mapillary_od_obstacle4_ground_materialize.yaml")
    )

    assert (
        balance["source_coco"]
        == "../work/datasets/obstacle4_ba_v2_hazard_ground_source/instances_normalized.json"
    )
    assert balance["source_images_dir"] == "../data/raw/obstacle4/extracted"
    assert balance["selection"]["max_positive_images_per_class"] == 100000
    assert [source["name"] for source in merge["sources"]] == [
        "ba_v2_hazard_slice01_mapillary_od",
        "obstacle4_ba_v2_ground_bootstrap",
    ]
    assert "images_dir" not in merge["sources"][0]
    assert "images_dir" not in merge["sources"][1]
    assert materialize["sources"][1]["images_dir"] == "../data/raw/obstacle4/extracted"
    assert materialize["sources"][1]["file_name_prefix"] == "obstacle4"


def test_ba_v2_mvp_training_config_points_to_materialized_candidate() -> None:
    cfg = _load_yaml(Path("configs/efficientdet_lite2_ba_v2_mvp.yaml"))

    assert cfg["model"]["variant"] == "lite2"
    assert cfg["data"]["csv"] == "work/datasets/ba_v2_mvp_candidate/modelmaker.csv"
    assert cfg["data"]["images_dir"] == "work/datasets/ba_v2_mvp_candidate/images"
    assert cfg["data"]["label_map_json"] == "configs/label_contracts/ba_v2_hazard.class_names.json"
    assert cfg["train"]["batch_size"] == 16
    assert cfg["train"]["epochs"] == 20


def test_mapillary_import_accepts_checked_in_ba_v2_hazard_slice(tmp_path: Path) -> None:
    source_root = tmp_path / "Map"
    _make_mapillary_stub(source_root)

    artifacts = import_mapillary_vistas_to_coco(
        mapillary_dir=source_root,
        out_dir=tmp_path / "out",
        label_map_path=Path("configs/label_maps/mapillary_vistas_to_ba_v2_hazard.yaml"),
        max_long_side=1600,
    )

    combined = json.loads(artifacts.combined_coco_path.read_text(encoding="utf-8"))
    observed_category_names = {
        category["name"]
        for category in combined["categories"]
        if any(ann["category_id"] == category["id"] for ann in combined["annotations"])
    }

    assert artifacts.categories == [
        "obstacle_barrier",
        "obstacle_hole_dropoff",
        "obstacle_pole",
        "person",
        "bicycle",
        "motorcycle",
        "car",
        "bus",
        "truck",
    ]
    assert [category["name"] for category in combined["categories"]] == artifacts.categories
    assert observed_category_names == {"obstacle_barrier", "person"}


def test_obstacle_dataset_import_accepts_checked_in_ba_v2_hazard_slice(tmp_path: Path) -> None:
    source_root = tmp_path / "Obstacle Dataset"
    _make_obstacle_dataset_stub(source_root)

    artifacts = import_obstacle_dataset_to_coco(
        dataset_dir=source_root,
        out_dir=tmp_path / "out",
        label_map_path=Path("configs/label_maps/obstacle_dataset_to_ba_v2_hazard.yaml"),
        mode="copy",
    )

    combined = json.loads(artifacts.combined_coco_path.read_text(encoding="utf-8"))

    assert artifacts.categories == ["obstacle_pole", "motorcycle", "car"]
    assert [category["name"] for category in combined["categories"]] == [
        "obstacle_pole",
        "motorcycle",
        "car",
    ]


def test_coco_replay_import_accepts_checked_in_ba_v2_hazard_slice(tmp_path: Path) -> None:
    config_path = _make_replay_fixture(tmp_path)

    artifacts = import_coco_replay_with_config(config_path)

    replay = json.loads(artifacts.coco_path.read_text(encoding="utf-8"))

    assert [category["name"] for category in replay["categories"]] == ["person", "car", "bus"]
