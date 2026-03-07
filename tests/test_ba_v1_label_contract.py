from pathlib import Path

import yaml


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_ba_v1_contract_has_fixed_unique_class_order():
    payload = _load_yaml(Path("configs/label_contracts/ba_v1.yaml"))

    assert payload["version"] == "ba_v1"
    assert payload["class_names"] == [
        "obstacle_bump",
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
    assert len(payload["class_names"]) == len(set(payload["class_names"]))


def test_ba_v1_roles_partition_classes_without_drift():
    payload = _load_yaml(Path("configs/label_contracts/ba_v1.yaml"))

    core = payload["roles"]["ba_core"]
    rehearsal = payload["roles"]["rehearsal"]

    assert core == [
        "obstacle_bump",
        "obstacle_fence",
        "obstacle_hole",
        "obstacle_pole",
    ]
    assert rehearsal == [
        "bicycle",
        "bus",
        "car",
        "motorcycle",
        "person",
        "truck",
    ]
    assert core + rehearsal == payload["class_names"]


def test_obstacle4_label_map_targets_only_ba_core_classes():
    contract = _load_yaml(Path("configs/label_contracts/ba_v1.yaml"))
    obstacle4_map = _load_yaml(Path("configs/label_maps/obstacle4_to_ba.yaml"))

    targets = set(obstacle4_map["map"].values())

    assert targets == set(contract["roles"]["ba_core"])
