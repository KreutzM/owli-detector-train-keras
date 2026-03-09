import json
from pathlib import Path

import yaml


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_ba_v2_hazard_contract_has_fixed_unique_class_order():
    payload = _load_yaml(Path("configs/label_contracts/ba_v2_hazard.yaml"))

    assert payload["version"] == "ba_v2_hazard"
    assert payload["class_names"] == [
        "obstacle_ground",
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
    assert len(payload["class_names"]) == len(set(payload["class_names"]))


def test_ba_v2_hazard_roles_partition_classes_without_drift():
    payload = _load_yaml(Path("configs/label_contracts/ba_v2_hazard.yaml"))

    hazard_core = payload["roles"]["hazard_core"]
    rehearsal = payload["roles"]["rehearsal"]

    assert hazard_core == [
        "obstacle_ground",
        "obstacle_barrier",
        "obstacle_hole_dropoff",
        "obstacle_pole",
    ]
    assert rehearsal == [
        "person",
        "bicycle",
        "motorcycle",
        "car",
        "bus",
        "truck",
    ]
    assert hazard_core + rehearsal == payload["class_names"]


def test_ba_v2_hazard_json_matches_yaml_contract_order():
    yaml_payload = _load_yaml(Path("configs/label_contracts/ba_v2_hazard.yaml"))
    json_payload = json.loads(
        Path("configs/label_contracts/ba_v2_hazard.class_names.json").read_text(encoding="utf-8")
    )

    assert json_payload["class_names"] == yaml_payload["class_names"]
    assert json_payload["category_ids"] == list(range(1, len(yaml_payload["class_names"]) + 1))


def test_ba_v2_hazard_explicitly_retires_old_obstacle4_shaped_labels():
    payload = _load_yaml(Path("configs/label_contracts/ba_v2_hazard.yaml"))

    assert payload["historical_ba_v1_labels_not_in_contract"] == [
        "obstacle_bump",
        "obstacle_fence",
        "obstacle_hole",
    ]
    assert payload["deferred_non_mvp_labels"] == ["obstacle_overhang"]
