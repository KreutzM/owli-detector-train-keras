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

    assert mapillary["status"] == "local_vistas_v1_2_verified"
    assert mapillary["drop_unmapped"] is True
    assert set(mapillary["map"].values()) <= set(contract["class_names"])
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
