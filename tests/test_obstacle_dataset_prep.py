from pathlib import Path

import yaml


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_obstacle_dataset_prep_is_constrained_to_ba_core() -> None:
    contract = _load_yaml(Path("configs/label_contracts/ba_v1.yaml"))
    prep = _load_yaml(Path("configs/label_maps/obstacle_dataset_to_ba.yaml"))

    assert prep["status"] == "pending_source_review"
    assert prep["allowed_target_classes"] == contract["roles"]["ba_core"]
    assert prep["drop_unmapped"] is True
    assert set(prep["map"].values()) <= set(contract["roles"]["ba_core"])


def test_obstacle_dataset_prep_starts_empty_until_source_taxonomy_is_verified() -> None:
    prep = _load_yaml(Path("configs/label_maps/obstacle_dataset_to_ba.yaml"))

    assert prep["map"] == {}
