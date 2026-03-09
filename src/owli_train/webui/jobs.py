from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from owli_train.webui.models import JobArtifactView, JobView
from owli_train.webui.readers import infer_repo_root

JOB_STATUS_VALUES = {"queued", "running", "succeeded", "failed"}
JOB_TYPE_TITLES = {
    "dataset_validate": "dataset validate",
    "dataset_split": "dataset split",
    "dataset_merge_coco": "dataset merge coco",
    "dataset_export_modelmaker_csv": "dataset export modelmaker-csv",
    "dataset_materialize_images": "dataset materialize-images",
}
JOB_LAUNCHER_ORDER = [
    "dataset_validate",
    "dataset_split",
    "dataset_merge_coco",
    "dataset_export_modelmaker_csv",
    "dataset_materialize_images",
]
JOB_MODE_OPTIONS = ("auto", "symlink", "copy")


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _code_root() -> Path:
    current = Path(__file__).resolve()
    return current.parents[3]


@dataclass(frozen=True)
class JobArtifact:
    label: str
    relative_path: str

    def to_payload(self) -> dict[str, str]:
        return {"label": self.label, "relative_path": self.relative_path}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> JobArtifact:
        return cls(
            label=str(payload.get("label", "")).strip(),
            relative_path=str(payload.get("relative_path", "")).strip(),
        )


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    job_type: str
    title: str
    status: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    exit_code: int | None
    command: list[str]
    command_preview: str
    parameters: dict[str, object]
    artifacts: list[JobArtifact]
    log_path: str
    pid: int | None = None
    error_message: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "title": self.title,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "command": self.command,
            "command_preview": self.command_preview,
            "parameters": self.parameters,
            "artifacts": [artifact.to_payload() for artifact in self.artifacts],
            "log_path": self.log_path,
            "pid": self.pid,
            "error_message": self.error_message,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> JobRecord:
        return cls(
            job_id=str(payload.get("job_id", "")).strip(),
            job_type=str(payload.get("job_type", "")).strip(),
            title=str(payload.get("title", "")).strip(),
            status=str(payload.get("status", "")).strip(),
            created_at=str(payload.get("created_at", "")).strip(),
            started_at=_optional_text(payload.get("started_at")),
            finished_at=_optional_text(payload.get("finished_at")),
            exit_code=_optional_int(payload.get("exit_code")),
            command=[str(item) for item in payload.get("command", [])],
            command_preview=str(payload.get("command_preview", "")).strip(),
            parameters=dict(payload.get("parameters", {})),
            artifacts=[
                JobArtifact.from_payload(item)
                for item in payload.get("artifacts", [])
                if isinstance(item, dict)
            ],
            log_path=str(payload.get("log_path", "")).strip(),
            pid=_optional_int(payload.get("pid")),
            error_message=_optional_text(payload.get("error_message")),
        )


@dataclass(frozen=True)
class JobFormCatalog:
    coco_files: list[str]
    image_dirs: list[str]
    merge_manifests: list[str]
    split_files: list[str]


@dataclass(frozen=True)
class JobLaunchResult:
    record: JobRecord


@dataclass(frozen=True)
class JobDetail:
    record: JobRecord
    log_text: str
    artifact_views: list[JobArtifactView]


class JobValidationError(ValueError):
    """Raised when submitted WebUI job parameters are invalid."""


class JobStore:
    def __init__(self, repo_root: Path | None = None):
        self.repo_root = infer_repo_root(repo_root)
        self.jobs_dir = self.repo_root / "work" / "webui" / "jobs"
        self.logs_dir = self.jobs_dir / "logs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def job_file_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def log_file_path(self, job_id: str) -> Path:
        return self.logs_dir / f"{job_id}.log"

    def create_job(self, record: JobRecord) -> JobRecord:
        self._write_record(record)
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        path = self.job_file_path(job_id)
        if not path.is_file():
            return None
        return JobRecord.from_payload(json.loads(path.read_text(encoding="utf-8")))

    def list_jobs(self, limit: int = 25) -> list[JobRecord]:
        jobs: list[JobRecord] = []
        for path in sorted(self.jobs_dir.glob("*.json")):
            jobs.append(JobRecord.from_payload(json.loads(path.read_text(encoding="utf-8"))))
        jobs.sort(key=lambda item: item.created_at, reverse=True)
        return jobs[:limit]

    def update_job(self, job_id: str, **changes: object) -> JobRecord:
        current = self.get_job(job_id)
        if current is None:
            raise FileNotFoundError(f"Unknown job id: {job_id}")
        payload = current.to_payload()
        payload.update(changes)
        updated = JobRecord.from_payload(payload)
        self._write_record(updated)
        return updated

    def _write_record(self, record: JobRecord) -> None:
        path = self.job_file_path(record.job_id)
        payload = json.dumps(record.to_payload(), indent=2)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(path)


class JobService:
    def __init__(self, repo_root: Path | None = None):
        self.repo_root = infer_repo_root(repo_root)
        self.store = JobStore(self.repo_root)
        self.code_root = _code_root()

    def list_jobs(self, limit: int = 25) -> list[JobView]:
        return [self._to_job_view(record) for record in self.store.list_jobs(limit=limit)]

    def get_job_detail(self, job_id: str) -> JobDetail | None:
        record = self.store.get_job(job_id)
        if record is None:
            return None
        log_abs_path = self.repo_root / record.log_path
        log_text = log_abs_path.read_text(encoding="utf-8") if log_abs_path.is_file() else ""
        artifact_views = [
            JobArtifactView(
                label=artifact.label,
                relative_path=artifact.relative_path,
                exists=(self.repo_root / artifact.relative_path).exists(),
            )
            for artifact in record.artifacts
        ]
        return JobDetail(record=record, log_text=log_text, artifact_views=artifact_views)

    def build_form_catalog(self) -> JobFormCatalog:
        return JobFormCatalog(
            coco_files=self._collect_file_choices(
                (
                    "tests/data/*.json",
                    "tests/smoke_coco/*.json",
                    "work/datasets/*/instances*.json",
                    "work/datasets/*/*/instances*.json",
                    "work/splits/*/instances*.json",
                )
            ),
            image_dirs=self._collect_dir_choices(
                (
                    "tests/smoke_coco/images",
                    "work/datasets/*/images",
                    "work/datasets/*/*/images",
                    "data/*/images",
                    "data/*/*/images",
                    "data/*/*/*/images",
                )
            ),
            merge_manifests=self._collect_file_choices(("configs/merge_*.yaml",)),
            split_files=self._collect_file_choices(
                ("work/splits/*/splits.json", "work/datasets/*/splits.json")
            ),
        )

    def default_form_values(self) -> dict[str, dict[str, str]]:
        catalog = self.build_form_catalog()
        first_coco = catalog.coco_files[0] if catalog.coco_files else ""
        first_images_dir = catalog.image_dirs[0] if catalog.image_dirs else ""
        first_manifest = catalog.merge_manifests[0] if catalog.merge_manifests else ""
        first_splits = catalog.split_files[0] if catalog.split_files else ""
        return {
            "dataset_validate": {
                "coco_path": first_coco,
                "images_dir": "",
            },
            "dataset_split": {
                "coco_path": first_coco,
                "out_dir": "",
                "seed": "1337",
                "write_coco": "",
                "ensure_train_class_coverage": "",
            },
            "dataset_merge_coco": {
                "manifest_path": first_manifest,
                "out_coco": "",
            },
            "dataset_export_modelmaker_csv": {
                "coco_path": first_coco,
                "images_dir": first_images_dir,
                "splits_json": first_splits,
                "out_csv": "",
            },
            "dataset_materialize_images": {
                "coco_path": first_coco,
                "merge_manifest": first_manifest,
                "out_images_dir": "",
                "out_coco": "",
                "mode": "auto",
                "overwrite": "",
            },
        }

    def start_job(self, job_type: str, form_data: dict[str, str]) -> JobLaunchResult:
        record = self._build_job_record(job_type=job_type, form_data=form_data)
        self.store.create_job(record)
        self._spawn_worker(record)
        return JobLaunchResult(record=record)

    def _build_job_record(self, *, job_type: str, form_data: dict[str, str]) -> JobRecord:
        if job_type not in JOB_TYPE_TITLES:
            raise JobValidationError(f"Unsupported job type: {job_type}")

        builders = {
            "dataset_validate": self._build_validate_job,
            "dataset_split": self._build_split_job,
            "dataset_merge_coco": self._build_merge_job,
            "dataset_export_modelmaker_csv": self._build_modelmaker_export_job,
            "dataset_materialize_images": self._build_materialize_job,
        }
        parameters, command, artifacts = builders[job_type](form_data)
        job_id = f"{datetime.now(tz=timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        log_rel_path = f"work/webui/jobs/logs/{job_id}.log"
        return JobRecord(
            job_id=job_id,
            job_type=job_type,
            title=JOB_TYPE_TITLES[job_type],
            status="queued",
            created_at=utc_now_iso(),
            started_at=None,
            finished_at=None,
            exit_code=None,
            command=command,
            command_preview=shlex.join(command),
            parameters=parameters,
            artifacts=artifacts,
            log_path=log_rel_path,
            pid=None,
            error_message=None,
        )

    def _build_validate_job(
        self, form_data: dict[str, str]
    ) -> tuple[dict[str, object], list[str], list[JobArtifact]]:
        catalog = self.build_form_catalog()
        coco_path = self._require_catalog_choice(
            form_data.get("coco_path", ""),
            catalog.coco_files,
            "COCO file",
        )
        images_dir = self._optional_catalog_choice(
            form_data.get("images_dir", ""),
            catalog.image_dirs,
            "Images directory",
        )
        parameters: dict[str, object] = {"coco": coco_path}
        command = [sys.executable, "-m", "owli_train", "dataset", "validate", "--coco", coco_path]
        if images_dir is not None:
            command.extend(["--images-dir", images_dir])
            parameters["images_dir"] = images_dir
        else:
            parameters["images_dir"] = None
        return parameters, command, []

    def _build_split_job(
        self, form_data: dict[str, str]
    ) -> tuple[dict[str, object], list[str], list[JobArtifact]]:
        catalog = self.build_form_catalog()
        coco_path = self._require_catalog_choice(
            form_data.get("coco_path", ""),
            catalog.coco_files,
            "COCO file",
        )
        seed = self._parse_positive_int(form_data.get("seed", "1337"), "Seed")
        write_coco = self._as_checkbox(form_data.get("write_coco"))
        ensure_coverage = self._as_checkbox(form_data.get("ensure_train_class_coverage"))
        default_out_dir = f"work/webui/splits/{Path(coco_path).stem}"
        out_dir = self._resolve_output_path(
            form_data.get("out_dir", ""),
            default_relative=default_out_dir,
            allow_file=False,
        )

        command = [
            sys.executable,
            "-m",
            "owli_train",
            "dataset",
            "split",
            "--coco",
            coco_path,
            "--out-dir",
            out_dir,
            "--seed",
            str(seed),
        ]
        if ensure_coverage:
            command.append("--ensure-train-class-coverage")
        if write_coco:
            command.append("--write-coco")

        artifacts = [JobArtifact(label="splits.json", relative_path=f"{out_dir}/splits.json")]
        if write_coco:
            artifacts.extend(
                [
                    JobArtifact(
                        label="instances_train.json",
                        relative_path=f"{out_dir}/instances_train.json",
                    ),
                    JobArtifact(
                        label="instances_val.json", relative_path=f"{out_dir}/instances_val.json"
                    ),
                    JobArtifact(
                        label="instances_test.json", relative_path=f"{out_dir}/instances_test.json"
                    ),
                ]
            )

        parameters = {
            "coco": coco_path,
            "out_dir": out_dir,
            "seed": seed,
            "write_coco": write_coco,
            "ensure_train_class_coverage": ensure_coverage,
        }
        return parameters, command, artifacts

    def _build_merge_job(
        self, form_data: dict[str, str]
    ) -> tuple[dict[str, object], list[str], list[JobArtifact]]:
        catalog = self.build_form_catalog()
        manifest_path = self._require_catalog_choice(
            form_data.get("manifest_path", ""),
            catalog.merge_manifests,
            "Merge manifest",
        )
        default_out = f"work/webui/merged/{Path(manifest_path).stem}.instances.json"
        out_coco = self._resolve_output_path(
            form_data.get("out_coco", ""),
            default_relative=default_out,
            allow_file=True,
        )
        report_out = str(Path(out_coco).with_suffix(".report.json"))

        command = [
            sys.executable,
            "-m",
            "owli_train",
            "dataset",
            "merge",
            "coco",
            "--manifest",
            manifest_path,
            "--out",
            out_coco,
            "--report-out",
            report_out,
        ]
        parameters = {"manifest": manifest_path, "out": out_coco, "report_out": report_out}
        artifacts = [
            JobArtifact(label="merged COCO", relative_path=out_coco),
            JobArtifact(label="merge report", relative_path=report_out),
        ]
        return parameters, command, artifacts

    def _build_modelmaker_export_job(
        self, form_data: dict[str, str]
    ) -> tuple[dict[str, object], list[str], list[JobArtifact]]:
        catalog = self.build_form_catalog()
        coco_path = self._require_catalog_choice(
            form_data.get("coco_path", ""),
            catalog.coco_files,
            "COCO file",
        )
        images_dir = self._require_catalog_choice(
            form_data.get("images_dir", ""),
            catalog.image_dirs,
            "Images directory",
        )
        splits_json = self._optional_catalog_choice(
            form_data.get("splits_json", ""),
            catalog.split_files,
            "Splits JSON",
        )
        default_out = f"work/webui/modelmaker/{Path(coco_path).stem}.csv"
        out_csv = self._resolve_output_path(
            form_data.get("out_csv", ""),
            default_relative=default_out,
            allow_file=True,
        )
        class_names_out = str(Path(out_csv).with_suffix(".class_names.json"))

        command = [
            sys.executable,
            "-m",
            "owli_train",
            "dataset",
            "export",
            "modelmaker-csv",
            "--coco",
            coco_path,
            "--images-dir",
            images_dir,
            "--out",
            out_csv,
        ]
        if splits_json is not None:
            command.extend(["--splits-json", splits_json])

        parameters: dict[str, object] = {
            "coco": coco_path,
            "images_dir": images_dir,
            "out_csv": out_csv,
            "splits_json": splits_json,
        }
        artifacts = [
            JobArtifact(label="modelmaker CSV", relative_path=out_csv),
            JobArtifact(label="class names JSON", relative_path=class_names_out),
        ]
        return parameters, command, artifacts

    def _build_materialize_job(
        self, form_data: dict[str, str]
    ) -> tuple[dict[str, object], list[str], list[JobArtifact]]:
        catalog = self.build_form_catalog()
        coco_path = self._require_catalog_choice(
            form_data.get("coco_path", ""),
            catalog.coco_files,
            "COCO file",
        )
        merge_manifest = self._require_catalog_choice(
            form_data.get("merge_manifest", ""),
            catalog.merge_manifests,
            "Merge manifest",
        )
        mode = str(form_data.get("mode", "auto")).strip() or "auto"
        if mode not in JOB_MODE_OPTIONS:
            raise JobValidationError("Materialize mode must be one of: auto, symlink, copy.")
        overwrite = self._as_checkbox(form_data.get("overwrite"))
        base_dir = f"work/webui/materialized/{Path(coco_path).stem}"
        out_images_dir = self._resolve_output_path(
            form_data.get("out_images_dir", ""),
            default_relative=f"{base_dir}/images",
            allow_file=False,
        )
        out_coco = self._resolve_output_path(
            form_data.get("out_coco", ""),
            default_relative=f"{base_dir}/instances_materialized.json",
            allow_file=True,
        )

        command = [
            sys.executable,
            "-m",
            "owli_train",
            "dataset",
            "materialize-images",
            "--coco",
            coco_path,
            "--merge-manifest",
            merge_manifest,
            "--out-images-dir",
            out_images_dir,
            "--out-coco",
            out_coco,
            "--mode",
            mode,
        ]
        if overwrite:
            command.append("--overwrite")

        parameters = {
            "coco": coco_path,
            "merge_manifest": merge_manifest,
            "out_images_dir": out_images_dir,
            "out_coco": out_coco,
            "mode": mode,
            "overwrite": overwrite,
        }
        artifacts = [
            JobArtifact(label="materialized images dir", relative_path=out_images_dir),
            JobArtifact(label="materialized COCO", relative_path=out_coco),
        ]
        return parameters, command, artifacts

    def _spawn_worker(self, record: JobRecord) -> None:
        worker_command = [
            sys.executable,
            "-m",
            "owli_train.webui.worker",
            "--repo-root",
            str(self.repo_root),
            "--job-file",
            str(self.store.job_file_path(record.job_id)),
        ]
        try:
            subprocess.Popen(
                worker_command,
                cwd=str(self.repo_root),
                env=build_subprocess_env(self.code_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            self.store.update_job(
                record.job_id,
                status="failed",
                finished_at=utc_now_iso(),
                exit_code=-1,
                error_message=str(exc),
            )
            raise JobValidationError(f"Failed to launch job worker: {exc}") from exc

    def _collect_file_choices(self, patterns: tuple[str, ...]) -> list[str]:
        matches: set[str] = set()
        for pattern in patterns:
            for path in self.repo_root.glob(pattern):
                if path.is_file():
                    matches.add(self._relative_path(path))
        return sorted(matches)

    def _collect_dir_choices(self, patterns: tuple[str, ...]) -> list[str]:
        matches: set[str] = set()
        for pattern in patterns:
            for path in self.repo_root.glob(pattern):
                if path.is_dir():
                    matches.add(self._relative_path(path))
        return sorted(matches)

    def _relative_path(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.repo_root))

    @staticmethod
    def _as_checkbox(raw: str | None) -> bool:
        return str(raw or "").strip().lower() in {"1", "true", "on", "yes"}

    @staticmethod
    def _parse_positive_int(raw: str, label: str) -> int:
        try:
            value = int(str(raw).strip())
        except ValueError as exc:
            raise JobValidationError(f"{label} must be an integer.") from exc
        if value <= 0:
            raise JobValidationError(f"{label} must be > 0.")
        return value

    @staticmethod
    def _normalize_relative_path(raw: str) -> PurePosixPath:
        text = str(raw).strip().replace("\\", "/")
        if not text:
            raise JobValidationError("Output path must not be empty.")
        rel_path = PurePosixPath(text)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise JobValidationError("Output paths must stay inside the repository.")
        if not rel_path.parts or rel_path.parts[0] != "work":
            raise JobValidationError("Output paths must stay under work/.")
        return rel_path

    def _resolve_output_path(self, raw: str, *, default_relative: str, allow_file: bool) -> str:
        chosen = raw.strip() or default_relative
        rel_path = self._normalize_relative_path(chosen)
        if allow_file and str(rel_path).endswith("/"):
            raise JobValidationError("Output file path must point to a file, not a directory.")
        return rel_path.as_posix()

    @staticmethod
    def _require_catalog_choice(raw: str, options: list[str], label: str) -> str:
        value = str(raw).strip()
        if not value:
            raise JobValidationError(f"{label} is required.")
        if value not in options:
            raise JobValidationError(f"{label} must be selected from the known repository paths.")
        return value

    @staticmethod
    def _optional_catalog_choice(raw: str, options: list[str], label: str) -> str | None:
        value = str(raw).strip()
        if not value:
            return None
        if value not in options:
            raise JobValidationError(f"{label} must be selected from the known repository paths.")
        return value

    def _to_job_view(self, record: JobRecord) -> JobView:
        return JobView(
            job_id=record.job_id,
            job_type=record.job_type,
            title=record.title,
            status=record.status,
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            exit_code=record.exit_code,
            command_preview=record.command_preview,
            log_path=record.log_path,
            parameters=record.parameters,
            artifacts=[
                JobArtifactView(
                    label=artifact.label,
                    relative_path=artifact.relative_path,
                    exists=(self.repo_root / artifact.relative_path).exists(),
                )
                for artifact in record.artifacts
            ],
            log_text=None,
        )


def build_subprocess_env(code_root: Path | None = None) -> dict[str, str]:
    resolved_code_root = _code_root() if code_root is None else Path(code_root).resolve()
    env = os.environ.copy()
    py_path_parts = [str(resolved_code_root), str(resolved_code_root / "src")]
    existing = env.get("PYTHONPATH", "")
    if existing:
        py_path_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(py_path_parts)
    return env


def run_job_file(*, repo_root: Path, job_file: Path) -> JobRecord:
    resolved_repo_root = infer_repo_root(repo_root)
    store = JobStore(resolved_repo_root)
    if not job_file.is_file():
        raise FileNotFoundError(f"Missing job file: {job_file}")
    record = JobRecord.from_payload(json.loads(job_file.read_text(encoding="utf-8")))
    if record.status not in JOB_STATUS_VALUES:
        raise ValueError(f"Unknown job status: {record.status}")

    log_abs_path = resolved_repo_root / record.log_path
    log_abs_path.parent.mkdir(parents=True, exist_ok=True)

    with log_abs_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"[webui-worker] queued_at={record.created_at}\n")
        log_handle.write(f"[webui-worker] repo_root={resolved_repo_root}\n")
        log_handle.write(f"[webui-worker] command={record.command_preview}\n\n")
        log_handle.flush()

        started_at = utc_now_iso()
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                record.command,
                cwd=str(resolved_repo_root),
                env=build_subprocess_env(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            store.update_job(
                record.job_id,
                status="running",
                started_at=started_at,
                pid=process.pid,
                error_message=None,
            )
            exit_code = process.wait()
            finished_at = utc_now_iso()
            status = "succeeded" if exit_code == 0 else "failed"
            log_handle.write(f"\n[webui-worker] exit_code={exit_code}\n")
            updated = store.update_job(
                record.job_id,
                status=status,
                finished_at=finished_at,
                exit_code=exit_code,
            )
            return updated
        except Exception as exc:  # pragma: no cover - defensive runtime path
            finished_at = utc_now_iso()
            log_handle.write("\n[webui-worker] exception while running job\n")
            log_handle.write(traceback.format_exc())
            updated = store.update_job(
                record.job_id,
                status="failed",
                started_at=started_at,
                finished_at=finished_at,
                exit_code=-1,
                error_message=str(exc),
                pid=process.pid if process is not None else None,
            )
            return updated


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
