from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

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
    app = FastAPI(
        title="Owli Control UI",
        description="Phase 2 visibility and small whitelisted job control over contracts, datasets, runs, and artifacts.",
        version="0.1.0",
    )
    app.state.repo_root = resolved_repo_root

    def _reader() -> RepositoryReader:
        return RepositoryReader(app.state.repo_root)

    def _jobs() -> JobService:
        return JobService(app.state.repo_root)

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
