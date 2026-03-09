import time

from fastapi.testclient import TestClient

from owli_train.webui.app import create_app
from owli_train.webui.jobs import JobRecord, JobService, JobStore, run_job_file, utc_now_iso
from tests.webui_test_utils import build_sample_repo


def _wait_for_terminal_state(
    service: JobService, job_id: str, timeout_seconds: float = 5.0
) -> JobRecord:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        detail = service.get_job_detail(job_id)
        assert detail is not None
        if detail.record.status in {"succeeded", "failed"}:
            return detail.record
        time.sleep(0.1)
    raise AssertionError(f"Job {job_id} did not reach a terminal state in time.")


def test_job_store_persists_job_records(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    store = JobStore(repo_root)
    record = JobRecord(
        job_id="job-1",
        job_type="dataset_validate",
        title="dataset validate",
        status="queued",
        created_at=utc_now_iso(),
        started_at=None,
        finished_at=None,
        exit_code=None,
        command=["python", "-m", "owli_train", "dataset", "validate"],
        command_preview="python -m owli_train dataset validate",
        parameters={"coco": "tests/data/coco_min.json"},
        artifacts=[],
        log_path="work/webui/jobs/logs/job-1.log",
    )

    store.create_job(record)
    updated = store.update_job("job-1", status="succeeded", exit_code=0)

    assert store.get_job("job-1") is not None
    assert updated.status == "succeeded"
    assert updated.exit_code == 0
    assert store.list_jobs(limit=5)[0].job_id == "job-1"


def test_run_job_file_executes_real_validate_job(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    service = JobService(repo_root)
    result = service.start_job(
        "dataset_validate",
        {
            "coco_path": "tests/data/coco_min.json",
            "images_dir": "tests/smoke_coco/images",
        },
    )

    record = _wait_for_terminal_state(service, result.record.job_id)
    detail = service.get_job_detail(record.job_id)

    assert record.status == "succeeded"
    assert record.exit_code == 0
    assert detail is not None
    assert "COCO: images=1" in detail.log_text


def test_run_job_file_direct_marks_split_job_success(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    service = JobService(repo_root)
    record = service._build_job_record(
        job_type="dataset_split",
        form_data={
            "coco_path": "tests/data/coco_min.json",
            "out_dir": "work/webui/splits/direct",
            "seed": "1337",
            "write_coco": "on",
            "ensure_train_class_coverage": "on",
        },
    )
    service.store.create_job(record)

    completed = run_job_file(
        repo_root=repo_root,
        job_file=service.store.job_file_path(record.job_id),
    )

    assert completed.status == "succeeded"
    assert completed.exit_code == 0
    assert (repo_root / "work" / "webui" / "splits" / "direct" / "splits.json").is_file()
    assert (repo_root / "work" / "webui" / "splits" / "direct" / "instances_train.json").is_file()


def test_job_service_builds_other_whitelisted_jobs_with_safe_outputs(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    service = JobService(repo_root)

    merge_job = service._build_job_record(
        job_type="dataset_merge_coco",
        form_data={
            "manifest_path": "configs/merge_demo.yaml",
            "out_coco": "",
        },
    )
    export_job = service._build_job_record(
        job_type="dataset_export_modelmaker_csv",
        form_data={
            "coco_path": "tests/data/coco_min.json",
            "images_dir": "tests/smoke_coco/images",
            "splits_json": "work/splits/demo-dataset/splits.json",
            "out_csv": "",
        },
    )
    materialize_job = service._build_job_record(
        job_type="dataset_materialize_images",
        form_data={
            "coco_path": "tests/data/coco_min.json",
            "merge_manifest": "configs/merge_demo.yaml",
            "out_images_dir": "",
            "out_coco": "",
            "mode": "symlink",
        },
    )

    assert "--manifest" in merge_job.command
    assert any(artifact.relative_path.endswith(".report.json") for artifact in merge_job.artifacts)
    assert "--splits-json" in export_job.command
    assert any(
        artifact.relative_path.endswith(".class_names.json") for artifact in export_job.artifacts
    )
    assert "work/webui/materialized" in materialize_job.command_preview
    assert any(artifact.label == "materialized COCO" for artifact in materialize_job.artifacts)


def test_job_routes_render_and_launch_validate_job(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    client = TestClient(create_app(repo_root=repo_root))

    jobs_page = client.get("/jobs")
    assert jobs_page.status_code == 200
    assert "dataset validate" in jobs_page.text
    assert "dataset materialize-images" in jobs_page.text

    launch = client.post(
        "/jobs/launch/dataset_validate",
        data={
            "coco_path": "tests/data/coco_min.json",
            "images_dir": "tests/smoke_coco/images",
        },
        follow_redirects=False,
    )
    assert launch.status_code == 303

    job_id = launch.headers["location"].rstrip("/").split("/")[-1]
    service = JobService(repo_root)
    record = _wait_for_terminal_state(service, job_id)
    detail = client.get(f"/jobs/{job_id}")

    assert record.status == "succeeded"
    assert detail.status_code == 200
    assert "COCO: images=1" in detail.text


def test_job_route_rejects_unknown_input_choice(tmp_path):
    repo_root = build_sample_repo(tmp_path)
    client = TestClient(create_app(repo_root=repo_root))

    response = client.post(
        "/jobs/launch/dataset_validate",
        data={
            "coco_path": "not/allowed.json",
            "images_dir": "",
        },
    )

    assert response.status_code == 400
    assert "must be selected from the known repository paths" in response.text
