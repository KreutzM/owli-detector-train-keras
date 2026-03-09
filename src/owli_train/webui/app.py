from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from owli_train.webui.fiftyone import FiftyOneService
from owli_train.webui.jobs import (
    JOB_LAUNCHER_ORDER,
    JOB_MODE_OPTIONS,
    JOB_TYPE_TITLES,
    JobService,
    JobValidationError,
)
from owli_train.webui.readers import RepositoryReader, infer_repo_root

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def create_app(repo_root: Path | None = None) -> FastAPI:
    resolved_repo_root = infer_repo_root(repo_root)
    fiftyone_service = FiftyOneService(resolved_repo_root)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            fiftyone_service.shutdown()

    app = FastAPI(
        title="Owli Control UI",
        description="Phase 4 visibility, diagnostics, small whitelisted job control, and a first local FiftyOne bridge for datasets and eval-linked datasets.",
        version="0.4.0",
        lifespan=lifespan,
    )
    app.state.repo_root = resolved_repo_root
    app.state.fiftyone = fiftyone_service

    def _reader() -> RepositoryReader:
        return RepositoryReader(app.state.repo_root)

    def _jobs() -> JobService:
        return JobService(app.state.repo_root)

    def _fiftyone() -> FiftyOneService:
        return app.state.fiftyone

    def _context(request: Request, page_title: str) -> dict[str, object]:
        return {
            "request": request,
            "page_title": page_title,
            "model": _reader().build_view_model(),
        }

    @app.get("/", response_class=HTMLResponse, name="dashboard")
    def dashboard(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "dashboard.html", _context(request, "Dashboard"))

    @app.get("/contracts", response_class=HTMLResponse, name="contracts_page")
    def contracts_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "contracts.html", _context(request, "Contracts"))

    @app.get("/artifacts", response_class=HTMLResponse, name="artifacts_page")
    def artifacts_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "artifacts.html", _context(request, "Artifacts"))

    @app.get("/datasets/view", response_class=HTMLResponse, name="dataset_detail_page")
    def dataset_detail_page(request: Request, path: str) -> HTMLResponse:
        detail = _reader().load_dataset_detail(path)
        if detail is None:
            raise HTTPException(status_code=404, detail="Unknown dataset path.")
        context = {
            **_context(request, "Dataset Detail"),
            "dataset": detail,
        }
        return templates.TemplateResponse(request, "dataset_detail.html", context)

    @app.get("/runs/view", response_class=HTMLResponse, name="run_detail_page")
    def run_detail_page(request: Request, path: str) -> HTMLResponse:
        detail = _reader().load_run_detail(path)
        if detail is None:
            raise HTTPException(status_code=404, detail="Unknown run path.")
        context = {
            **_context(request, "Run Detail"),
            "run": detail,
        }
        return templates.TemplateResponse(request, "run_detail.html", context)

    @app.get("/evals/view", response_class=HTMLResponse, name="eval_detail_page")
    def eval_detail_page(request: Request, path: str) -> HTMLResponse:
        detail = _reader().load_eval_detail(path)
        if detail is None:
            raise HTTPException(status_code=404, detail="Unknown eval report path.")
        context = {
            **_context(request, "Eval Detail"),
            "eval_detail": detail,
        }
        return templates.TemplateResponse(request, "eval_detail.html", context)

    @app.get("/goldens/view", response_class=HTMLResponse, name="golden_detail_page")
    def golden_detail_page(request: Request, path: str) -> HTMLResponse:
        detail = _reader().load_golden_detail(path)
        if detail is None:
            raise HTTPException(status_code=404, detail="Unknown golden report path.")
        context = {
            **_context(request, "Golden Detail"),
            "golden": detail,
        }
        return templates.TemplateResponse(request, "golden_detail.html", context)

    @app.get("/fiftyone/open", response_class=HTMLResponse, name="fiftyone_launch_page")
    def fiftyone_launch_page(request: Request, source: str, path: str) -> HTMLResponse:
        target = _reader().resolve_fiftyone_target(source_kind=source, relative_path=path)
        if target is None:
            raise HTTPException(status_code=404, detail="Unknown FiftyOne source path.")
        if not target.can_launch:
            context = {
                **_context(request, "FiftyOne"),
                "launch_target": target,
                "launch_result": _fiftyone().launch(target),
            }
            return templates.TemplateResponse(
                request,
                "fiftyone_launch.html",
                context,
                status_code=400,
            )
        launch_result = _fiftyone().launch(target)
        context = {
            **_context(request, "FiftyOne"),
            "launch_target": target,
            "launch_result": launch_result,
        }
        return templates.TemplateResponse(
            request,
            "fiftyone_launch.html",
            context,
            status_code=200 if launch_result.status == "ready" else 503,
        )

    def _jobs_context(
        request: Request,
        *,
        page_title: str,
        form_values: dict[str, dict[str, str]] | None = None,
        form_errors: dict[str, list[str]] | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        jobs_service = _jobs()
        context = {
            **_context(request, page_title),
            "jobs": jobs_service.list_jobs(),
            "job_catalog": jobs_service.build_form_catalog(),
            "job_values": form_values or jobs_service.default_form_values(),
            "job_errors": form_errors or {},
            "job_launcher_order": JOB_LAUNCHER_ORDER,
            "job_titles": JOB_TYPE_TITLES,
            "job_mode_options": JOB_MODE_OPTIONS,
        }
        return templates.TemplateResponse(request, "jobs.html", context, status_code=status_code)

    @app.get("/jobs", response_class=HTMLResponse, name="jobs_page")
    def jobs_page(request: Request) -> HTMLResponse:
        return _jobs_context(request, page_title="Jobs")

    @app.post("/jobs/launch/{job_type}", name="launch_job")
    async def launch_job(request: Request, job_type: str):
        submitted = await request.form()
        form_data = {str(key): str(value) for key, value in submitted.items()}
        job_service = _jobs()
        form_values = job_service.default_form_values()
        form_values.setdefault(job_type, {}).update(form_data)
        try:
            result = job_service.start_job(job_type=job_type, form_data=form_data)
        except JobValidationError as exc:
            return _jobs_context(
                request,
                page_title="Jobs",
                form_values=form_values,
                form_errors={job_type: [str(exc)]},
                status_code=400,
            )
        return RedirectResponse(
            request.url_for("job_detail_page", job_id=result.record.job_id),
            status_code=303,
        )

    @app.get("/jobs/{job_id}", response_class=HTMLResponse, name="job_detail_page")
    def job_detail_page(request: Request, job_id: str) -> HTMLResponse:
        jobs_service = _jobs()
        detail = jobs_service.get_job_detail(job_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Unknown job id.")
        context = {
            **_context(request, "Job Detail"),
            "job": detail.record,
            "job_log_text": detail.log_text,
            "job_artifacts": detail.artifact_views,
        }
        return templates.TemplateResponse(request, "job_detail.html", context)

    return app


app = create_app()
