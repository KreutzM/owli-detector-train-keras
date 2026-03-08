from pathlib import Path

import yaml


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_mapillary_and_taco_prep_stay_constrained_to_ba_core() -> None:
    contract = _load_yaml(Path("configs/label_contracts/ba_v1.yaml"))
    expected_targets = contract["roles"]["ba_core"]

    for rel_path in (
        "configs/label_maps/mapillary_vistas_to_ba.yaml",
        "configs/label_maps/taco_to_ba.yaml",
    ):
        prep = _load_yaml(Path(rel_path))
        assert prep["status"] == "pending_source_review"
        assert prep["allowed_target_classes"] == expected_targets
        assert prep["drop_unmapped"] is True
        assert prep["map"] == {}


def test_coco_replay_prep_stays_narrow_and_identity_mapped() -> None:
    contract = _load_yaml(Path("configs/label_contracts/ba_v1.yaml"))
    replay = _load_yaml(Path("configs/label_maps/coco_replay_to_ba.yaml"))

    expected_rehearsal = contract["roles"]["rehearsal"]

    assert replay["status"] == "replay_subset_prep"
    assert replay["allowed_target_classes"] == expected_rehearsal
    assert replay["drop_unmapped"] is True
    assert replay["map"] == {name: name for name in expected_rehearsal}
