from owli_train.webui.readers import RepositoryReader
from tests.webui_test_utils import build_sample_repo


def test_repository_reader_loads_contracts_and_roles(tmp_path):
    repo_root = build_sample_repo(tmp_path)

    reader = RepositoryReader(repo_root)
    contracts = reader.load_contracts()

    assert [contract.key for contract in contracts[:2]] == ["ba_v1", "ba_v2_hazard"]
    assert contracts[0].classes[0].name == "obstacle_bump"
    assert contracts[0].roles[0].name == "ba_core"
    assert contracts[1].roles[0].class_names == ["obstacle_ground"]


def test_repository_reader_detects_artifacts_and_config_targets(tmp_path):
    repo_root = build_sample_repo(tmp_path)

    reader = RepositoryReader(repo_root)
    model = reader.build_view_model()

    assert any(
        item.relative_path == "work/datasets" and item.exists for item in model.artifact_roots
    )
    assert model.recent_datasets[0].relative_path == "work/datasets/demo-dataset"
    assert model.recent_runs[0].relative_path == "work/runs/20260309-123000-demo"
    assert any(
        group.title == "Dataset prep configs" and group.items for group in model.config_groups
    )
    assert any(group.title == "Training configs" and group.items for group in model.config_groups)
