from pathlib import Path

from owli_train.webui.fiftyone import FiftyOneService
from owli_train.webui.models import FiftyOneLaunchTargetView


def test_fiftyone_service_reports_missing_dependency(monkeypatch, tmp_path: Path):
    repo_root = tmp_path
    service = FiftyOneService(repo_root)
    target = FiftyOneLaunchTargetView(
        source_kind="dataset",
        source_path="work/datasets/demo-dataset",
        source_label="Dataset detail",
        back_path="work/datasets/demo-dataset",
        back_label="Back to dataset detail",
        back_route_name="dataset_detail_page",
        title="Open demo dataset in FiftyOne",
        dataset_name="owli-demo-dataset",
        coco_path="work/datasets/demo-dataset/instances_materialized.json",
        images_dir="work/datasets/demo-dataset/images",
        can_launch=True,
        message="Ready",
    )
    monkeypatch.setattr("owli_train.webui.fiftyone.importlib.util.find_spec", lambda _: None)

    result = service.launch(target)

    assert result.status == "error"
    assert "requirements/fiftyone.txt" in result.message
