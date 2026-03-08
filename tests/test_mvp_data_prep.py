from pathlib import Path

import yaml


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_mapillary_and_taco_prep_stay_constrained_to_ba_core() -> None:
    contract = _load_yaml(Path("configs/label_contracts/ba_v1.yaml"))
    taco = _load_yaml(Path("configs/label_maps/taco_to_ba.yaml"))

    assert taco["status"] == "pending_source_review"
    assert taco["allowed_target_classes"] == contract["roles"]["ba_core"]
    assert taco["drop_unmapped"] is True
    assert taco["map"] == {}


def test_mapillary_prep_maps_only_into_the_current_ba_v1_contract() -> None:
    contract = _load_yaml(Path("configs/label_contracts/ba_v1.yaml"))
    mapillary = _load_yaml(Path("configs/label_maps/mapillary_vistas_to_ba.yaml"))

    assert mapillary["status"] == "local_vistas_v1_2_and_v2_0_verified"
    assert mapillary["drop_unmapped"] is True
    assert set(mapillary["map"].values()) <= set(contract["class_names"])
    assert mapillary["map"]["human--person--individual"] == "person"
    assert "object--manhole" not in mapillary["map"]
    assert "human--rider--bicyclist" not in mapillary["map"]
    assert "human--rider--motorcyclist" not in mapillary["map"]


def test_coco_replay_prep_stays_narrow_and_identity_mapped() -> None:
    contract = _load_yaml(Path("configs/label_contracts/ba_v1.yaml"))
    replay = _load_yaml(Path("configs/label_maps/coco_replay_to_ba.yaml"))

    expected_rehearsal = contract["roles"]["rehearsal"]

    assert replay["status"] == "replay_subset_prep"
    assert replay["allowed_target_classes"] == expected_rehearsal
    assert replay["drop_unmapped"] is True
    assert replay["map"] == {name: name for name in expected_rehearsal}


def test_coco_replay_stage4_config_stays_small_and_points_to_clean_train2017() -> None:
    config = _load_yaml(Path("configs/coco_replay_ba_mvp_stage4.yaml"))

    assert config["status"] == "first_stage4_rehearsal_subset"
    assert config["source_coco"] == "../work/datasets/coco2017/instances_train2017.clean.json"
    assert config["source_images_dir"] == "../data/coco2017/images/train2017"
    assert config["label_map"] == "label_maps/coco_replay_to_ba.yaml"
    assert config["out_dir"] == "../work/datasets/coco_replay_ba_v1_stage4"
    assert config["selection"]["min_bbox_min_side"] == 16
    assert config["selection"]["max_positive_images_per_class"] == 250


def test_stage4_manifest_adds_coco_replay_as_a_fourth_source() -> None:
    manifest = _load_yaml(Path("configs/merge_ba_mvp_stage4_with_coco_replay.yaml"))

    assert [source["name"] for source in manifest["sources"]] == [
        "obstacle4_combined",
        "mapillary_vistas_ba_v1_mvp_balanced",
        "od_ba_v1",
        "coco_replay_ba_v1_stage4",
    ]
    assert (
        manifest["sources"][3]["coco"]
        == "../work/datasets/coco_replay_ba_v1_stage4/instances_ba_v1.coco.json"
    )
    assert manifest["sources"][3]["images_dir"] == "../data/coco2017/images/train2017"
    assert manifest["sources"][3]["file_name_prefix"] == "coco_replay"
    assert manifest["settings"]["allow_duplicate_file_names"] is False


def test_stage2_obstacle4_mapillary_manifest_stays_concrete_and_prefixed() -> None:
    manifest = _load_yaml(Path("configs/merge_ba_mvp_stage2_obstacle4_mapillary.yaml"))

    assert [source["name"] for source in manifest["sources"]] == [
        "obstacle4_combined",
        "mapillary_vistas_ba_v1",
    ]
    assert manifest["sources"][0]["coco"] == "../work/datasets/obstacle4/instances_combined.json"
    assert manifest["sources"][0]["file_name_prefix"] == "obstacle4"
    assert (
        manifest["sources"][1]["coco"]
        == "../work/datasets/mapillary_vistas_ba_v1/instances_ba_v1.coco.json"
    )
    assert manifest["sources"][1]["file_name_prefix"] == "mapillary_vistas"
    assert manifest["settings"]["allow_duplicate_file_names"] is False


def test_stage2_obstacle4_od_manifest_stays_concrete_and_prefixed() -> None:
    manifest = _load_yaml(Path("configs/merge_ba_mvp_stage2_obstacle4_od.yaml"))

    assert [source["name"] for source in manifest["sources"]] == [
        "obstacle4_combined",
        "od_ba_v1",
    ]
    assert manifest["sources"][0]["coco"] == "../work/datasets/obstacle4/instances_combined.json"
    assert manifest["sources"][0]["file_name_prefix"] == "obstacle4"
    assert manifest["sources"][1]["coco"] == "../work/datasets/od_ba_v1/instances_ba_v1.coco.json"
    assert manifest["sources"][1]["file_name_prefix"] == "od_ba_v1"
    assert manifest["settings"]["allow_duplicate_file_names"] is False


def test_mapillary_balance_config_stays_on_the_verified_v1_export() -> None:
    balance = _load_yaml(Path("configs/balance_ba_mvp_mapillary.yaml"))

    assert balance["status"] == "first_balanced_multisource_mvp"
    assert (
        balance["source_coco"]
        == "../work/datasets/mapillary_vistas_ba_v1/instances_ba_v1.coco.json"
    )
    assert balance["source_images_dir"] == "../work/datasets/mapillary_vistas_ba_v1/images"
    assert balance["source_splits_json"] == "../work/datasets/mapillary_vistas_ba_v1/splits.json"
    assert balance["out_dir"] == "../work/datasets/mapillary_vistas_ba_v1_mvp_balanced"
    assert balance["selection"]["min_bbox_min_side"] == 16
    assert balance["selection"]["max_positive_images_per_class"] == 400


def test_stage3_balanced_multisource_manifest_stays_concrete_and_prefixed() -> None:
    manifest = _load_yaml(Path("configs/merge_ba_mvp_stage3_balanced_multisource.yaml"))

    assert [source["name"] for source in manifest["sources"]] == [
        "obstacle4_combined",
        "mapillary_vistas_ba_v1_mvp_balanced",
        "od_ba_v1",
    ]
    assert manifest["sources"][0]["coco"] == "../work/datasets/obstacle4/instances_combined.json"
    assert (
        manifest["sources"][1]["coco"]
        == "../work/datasets/mapillary_vistas_ba_v1_mvp_balanced/instances_ba_v1.coco.json"
    )
    assert manifest["sources"][1]["images_dir"] == "../work/datasets/mapillary_vistas_ba_v1/images"
    assert manifest["sources"][1]["file_name_prefix"] == "mapillary_vistas"
    assert manifest["sources"][2]["coco"] == "../work/datasets/od_ba_v1/instances_ba_v1.coco.json"
    assert manifest["sources"][2]["file_name_prefix"] == "od_ba_v1"
    assert manifest["settings"]["allow_duplicate_file_names"] is False
