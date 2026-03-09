import json

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
    assert any(item.relative_path == "work/runs/20260309-123000-demo" for item in model.recent_runs)
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
    assert dataset.fiftyone_target is not None
    assert dataset.fiftyone_target.can_launch is True
    assert dataset.fiftyone_target.images_dir == "work/datasets/demo-dataset/images"

    assert run is not None
    assert any(item.label == "eval_demo.json" for item in run.report_files)
    assert run.eval_reports[0].note == "mAP50=0.75"

    assert eval_detail is not None
    assert eval_detail.fiftyone_target is not None
    assert eval_detail.fiftyone_target.can_launch is True
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


def test_repository_reader_marks_dataset_without_images_dir_as_not_launchable(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    images_dir = repo_root / "work" / "datasets" / "demo-dataset" / "images"
    for path in images_dir.iterdir():
        path.unlink()
    images_dir.rmdir()

    reader = RepositoryReader(repo_root)
    dataset = reader.load_dataset_detail("work/datasets/demo-dataset")

    assert dataset is not None
    assert dataset.fiftyone_target is not None
    assert dataset.fiftyone_target.can_launch is False
    assert "no local images directory" in dataset.fiftyone_target.message.lower()


def test_repository_reader_builds_run_compare_from_common_eval_target(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    reader = RepositoryReader(repo_root)

    compare_view = reader.load_runs_compare()

    assert compare_view.selected_target_label == "ba_mvp_stage3_balanced_multisource / TEST"
    assert [row.run_display_name for row in compare_view.rows] == [
        "Stage-3 baseline",
        "Stage-4 replay baseline",
        "Stage-3-plus-crops baseline",
    ]
    assert compare_view.rows[0].config_path == "configs/efficientdet_lite2_ba_mvp_stage3.yaml"
    assert compare_view.rows[1].metrics["AP50"] == "0.229"
    assert compare_view.rows[2].golden_relative_path is not None
    assert compare_view.class_scope_key == "ba_core"
    assert compare_view.per_class_rows[0].label == "obstacle_bump"
    assert compare_view.per_class_rows[1].label == "obstacle_fence / obstacle_fence_rail"
    assert compare_view.per_class_rows[1].cells[1].matched_class_name == "obstacle_fence_rail"
    assert compare_view.per_class_rows[2].cells[2].matched_class_name == "obstacle_hole_dropoff"
    assert compare_view.per_class_rows[3].cells[0].metrics["tp"] == "163"


def test_repository_reader_compare_supports_ba_core_plus_rehearsal_scope(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    reader = RepositoryReader(repo_root)

    compare_view = reader.load_runs_compare(class_scope_key="ba_core_rehearsal")

    labels = [row.label for row in compare_view.per_class_rows]
    assert compare_view.class_scope_key == "ba_core_rehearsal"
    assert "person" in labels
    assert "car" in labels
    assert "truck" in labels
    person_row = next(row for row in compare_view.per_class_rows if row.label == "person")
    assert person_row.cells[0].metrics["precision"] == "0.2304"
    assert person_row.cells[2].metrics["recall"] == "0.4748"


def test_repository_reader_compare_handles_selected_run_and_missing_metrics(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    report_path = (
        repo_root / "work" / "runs" / "20260309-123000-demo" / "reports" / "eval_demo.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["metrics"] = {}
    report["summary_counts"] = {"tp": 1, "fp": 0, "fn": 0}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    reader = RepositoryReader(repo_root)
    compare_view = reader.load_runs_compare(selected_run_paths=["work/runs/20260309-123000-demo"])

    assert len(compare_view.rows) == 1
    assert compare_view.rows[0].run_relative_path == "work/runs/20260309-123000-demo"
    assert compare_view.rows[0].metrics["AP"] == "-"
    assert compare_view.rows[0].metrics["precision"] == "-"


def test_repository_reader_compare_handles_missing_per_class_payload(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    report_path = (
        repo_root
        / "work"
        / "runs"
        / "20260308-183140-ba-mvp-stage3-20260308"
        / "reports"
        / "eval_efficientdet_tflite_stage3_test.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["per_class"] = {}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    reader = RepositoryReader(repo_root)
    compare_view = reader.load_runs_compare(
        selected_run_paths=["work/runs/20260308-183140-ba-mvp-stage3-20260308"]
    )

    assert compare_view.rows[0].run_display_name == "Stage-3 baseline"
    assert compare_view.per_class_rows == []
