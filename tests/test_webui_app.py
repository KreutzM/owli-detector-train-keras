from fastapi.testclient import TestClient

from owli_train.webui.app import create_app
from tests.webui_test_utils import build_sample_repo


def test_webui_routes_render_dashboard_contracts_and_artifacts(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    client = TestClient(create_app(repo_root=repo_root))

    dashboard = client.get("/")
    contracts = client.get("/contracts")
    artifacts = client.get("/artifacts")
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

    assert jobs.status_code == 200
    assert "dataset validate" in jobs.text

    assert dataset_detail.status_code == 200
    assert "Class distribution" in dataset_detail.text
    assert "small_bbox_filtered" in dataset_detail.text

    assert run_detail.status_code == 200
    assert "Eval reports" in run_detail.text
    assert "golden_obstacle4.json" in run_detail.text

    assert eval_detail.status_code == 200
    assert "Per-class metrics" in eval_detail.text
    assert "eval_demo_alt.json" in eval_detail.text

    assert golden_detail.status_code == 200
    assert "Detections" in golden_detail.text
    assert "demo-model" in golden_detail.text


def test_webui_detail_routes_return_404_for_unknown_paths(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    client = TestClient(create_app(repo_root=repo_root))

    response = client.get("/datasets/view", params={"path": "../outside"})
    assert response.status_code == 404
