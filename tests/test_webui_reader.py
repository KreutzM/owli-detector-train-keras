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


def test_repository_reader_builds_dataset_run_eval_and_golden_details(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    reader = RepositoryReader(repo_root)

    dataset = reader.load_dataset_detail("work/datasets/demo-dataset")
    run = reader.load_run_detail("work/runs/20260309-123000-demo")
    eval_detail = reader.load_eval_detail("work/runs/20260309-123000-demo/reports/eval_demo.json")
    golden = reader.load_golden_detail(
        "work/runs/20260309-123000-demo/reports/golden_obstacle4.json"
    )

    assert dataset is not None
    assert dataset.class_distribution[0].label == "person"
    assert dataset.split_counts[0].label == "train"
    assert any(item.label == "images" and item.value == "1" for item in dataset.qc_summary)

    assert run is not None
    assert any(item.label == "eval_demo.json" for item in run.report_files)
    assert run.eval_reports[0].note == "mAP50=0.75"

    assert eval_detail is not None
    assert eval_detail.metrics[0].key == "mAP50"
    assert eval_detail.per_class_rows[0].class_name == "person"
    assert eval_detail.sibling_reports[0].label == "eval_demo_alt.json"

    assert golden is not None
    assert golden.detections[0].class_name == "person"
    assert golden.contract[0].label == "class_labels_source"


def test_repository_reader_rejects_invalid_detail_paths(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    reader = RepositoryReader(repo_root)

    assert reader.load_dataset_detail("../outside") is None
    assert reader.load_run_detail("work/datasets/demo-dataset") is None
    assert reader.load_eval_detail("README.md") is None
    assert reader.load_golden_detail("work/runs/20260309-123000-demo") is None
