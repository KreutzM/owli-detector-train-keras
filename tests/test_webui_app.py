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

    assert dashboard.status_code == 200
    assert "Owli Control UI" in dashboard.text
    assert "work/datasets/demo-dataset" in dashboard.text

    assert contracts.status_code == 200
    assert "obstacle_ground" in contracts.text
    assert "preferred_product_target_pre_training_reset" in contracts.text

    assert artifacts.status_code == 200
    assert "efficientdet_demo.yaml" in artifacts.text
    assert "model.tflite" in artifacts.text

    assert jobs.status_code == 200
    assert "dataset validate" in jobs.text
