from pathlib import Path

import yaml


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_obstacle_dataset_prep_stays_inside_the_current_ba_v1_contract() -> None:
    contract = _load_yaml(Path("configs/label_contracts/ba_v1.yaml"))
    prep = _load_yaml(Path("configs/label_maps/obstacle_dataset_to_ba.yaml"))

    assert prep["status"] == "local_source_reviewed_partial_mapping"
    assert set(prep["allowed_target_classes"]) <= set(contract["class_names"])
    assert prep["drop_unmapped"] is True
    assert set(prep["map"].values()) <= set(contract["class_names"])
    assert prep["map"]["pole"] == "obstacle_pole"
    assert prep["map"]["warning_column"] == "obstacle_pole"
    assert prep["map"]["motorbike"] == "motorcycle"


def test_obstacle_dataset_prep_keeps_uncertain_classes_unmapped() -> None:
    prep = _load_yaml(Path("configs/label_maps/obstacle_dataset_to_ba.yaml"))

    assert "reflective_cone" not in prep["map"]
    assert "spherical_roadblock" not in prep["map"]
    assert "ashcan" not in prep["map"]
    assert "fire_hydrant" not in prep["map"]
