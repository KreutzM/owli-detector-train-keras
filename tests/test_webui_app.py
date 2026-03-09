from fastapi.testclient import TestClient

from owli_train.webui.app import create_app
from owli_train.webui.models import FiftyOneLaunchResultView
from tests.webui_test_utils import build_sample_repo


class FakeFiftyOneService:
    def __init__(self, result: FiftyOneLaunchResultView):
        self.result = result

    def launch(self, target):
        return self.result

    def shutdown(self):
        return None


def test_webui_routes_render_dashboard_contracts_and_artifacts(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    client = TestClient(create_app(repo_root=repo_root))

    dashboard = client.get("/")
    contracts = client.get("/contracts")
    artifacts = client.get("/artifacts")
    compare_runs = client.get("/compare/runs")
    jobs = client.get("/jobs")
    dataset_detail = client.get("/datasets/view", params={"path": "work/datasets/demo-dataset"})
    run_detail = client.get("/runs/view", params={"path": "work/runs/20260309-123000-demo"})
    eval_detail = client.get(
        "/evals/view",
        params={"path": "work/runs/20260309-123000-demo/reports/eval_demo.json"},
    )
    golden_detail = client.get(
        "/goldens/view",
        params={"path": "work/runs/20260309-123000-demo/reports/golden_obstacle4.json"},
    )

    assert dashboard.status_code == 200
    assert "Owli Control UI" in dashboard.text
    assert "work/datasets/demo-dataset" in dashboard.text
    assert "Open dataset detail" in dashboard.text

    assert contracts.status_code == 200
    assert "obstacle_ground" in contracts.text
    assert "preferred_product_target_pre_training_reset" in contracts.text

    assert artifacts.status_code == 200
    assert "efficientdet_demo.yaml" in artifacts.text
    assert "model.tflite" in artifacts.text

    assert compare_runs.status_code == 200
    assert "Run / Eval Compare" in compare_runs.text
    assert "Stage-3 baseline" in compare_runs.text
    assert "ba_mvp_stage3_balanced_multisource / TEST" in compare_runs.text
    assert "Baseline reference:" in compare_runs.text
    assert "Delta AP50" in compare_runs.text
    assert "baseline" in compare_runs.text
    assert "Curated per-class view" in compare_runs.text
    assert "obstacle_fence / obstacle_fence_rail" in compare_runs.text
    assert "using <code>obstacle_fence_rail</code>" in compare_runs.text

    assert jobs.status_code == 200
    assert "dataset validate" in jobs.text

    assert dataset_detail.status_code == 200
    assert "Class distribution" in dataset_detail.text
    assert "small_bbox_filtered" in dataset_detail.text
    assert "Open in FiftyOne" in dataset_detail.text

    assert run_detail.status_code == 200
    assert "Eval reports" in run_detail.text
    assert "golden_obstacle4.json" in run_detail.text

    assert eval_detail.status_code == 200
    assert "Per-class metrics" in eval_detail.text
    assert "eval_demo_alt.json" in eval_detail.text
    assert "Open eval dataset in FiftyOne" in eval_detail.text

    assert golden_detail.status_code == 200
    assert "Detections" in golden_detail.text
    assert "demo-model" in golden_detail.text


def test_webui_detail_routes_return_404_for_unknown_paths(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    client = TestClient(create_app(repo_root=repo_root))

    response = client.get("/datasets/view", params={"path": "../outside"})
    assert response.status_code == 404


def test_compare_runs_route_supports_run_filter_and_target_selection(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    client = TestClient(create_app(repo_root=repo_root))

    response = client.get(
        "/compare/runs",
        params=[
            ("run", "work/runs/20260308-211806-ba-mvp-stage4-20260308"),
            ("target", "split:ba_mvp_stage4_with_coco_replay:test"),
        ],
    )

    assert response.status_code == 200
    assert "Stage-4 replay baseline" in response.text
    assert "Showing 1 eval rows across 1 runs" in response.text
    assert "0.24" in response.text


def test_compare_runs_route_supports_per_class_scope_switch(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    client = TestClient(create_app(repo_root=repo_root))

    response = client.get(
        "/compare/runs",
        params={"class_scope": "ba_core_rehearsal"},
    )

    assert response.status_code == 200
    assert "BA core + rehearsal" in response.text
    assert "person" in response.text
    assert "bicycle" in response.text
    assert "truck" in response.text


def test_compare_runs_route_supports_explicit_baseline_selection(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    client = TestClient(create_app(repo_root=repo_root))

    response = client.get(
        "/compare/runs",
        params={"baseline": "work/runs/20260308-211806-ba-mvp-stage4-20260308"},
    )

    assert response.status_code == 200
    assert "Baseline reference: <strong>Stage-4 replay baseline</strong>." in response.text
    assert "Delta AP50" in response.text
    assert "+0.0035" in response.text


def test_fiftyone_launch_route_renders_ready_state_with_local_link(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    app = create_app(repo_root=repo_root)
    app.state.fiftyone = FakeFiftyOneService(
        FiftyOneLaunchResultView(
            status="ready",
            message="Started local session.",
            app_url="http://127.0.0.1:5151/",
        )
    )
    client = TestClient(app)

    response = client.get(
        "/fiftyone/open",
        params={"source": "dataset", "path": "work/datasets/demo-dataset"},
    )

    assert response.status_code == 200
    assert "Started local session." in response.text
    assert "http://127.0.0.1:5151/" in response.text


def test_fiftyone_launch_route_returns_503_for_launch_failure(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    app = create_app(repo_root=repo_root)
    app.state.fiftyone = FakeFiftyOneService(
        FiftyOneLaunchResultView(
            status="error",
            message="FiftyOne is not installed in this venv.",
        )
    )
    client = TestClient(app)

    response = client.get(
        "/fiftyone/open",
        params={"source": "dataset", "path": "work/datasets/demo-dataset"},
    )

    assert response.status_code == 503
    assert "FiftyOne is not installed in this venv." in response.text


def test_fiftyone_launch_route_renders_clear_error_for_unlaunchable_dataset(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    images_dir = repo_root / "work" / "datasets" / "demo-dataset" / "images"
    for path in images_dir.iterdir():
        path.unlink()
    images_dir.rmdir()

    client = TestClient(create_app(repo_root=repo_root))
    response = client.get(
        "/fiftyone/open",
        params={"source": "dataset", "path": "work/datasets/demo-dataset"},
    )

    assert response.status_code == 400
    assert "not ready for FiftyOne yet" in response.text
