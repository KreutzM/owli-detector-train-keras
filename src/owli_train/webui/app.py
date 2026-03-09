from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from owli_train.webui.readers import RepositoryReader, infer_repo_root

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def create_app(repo_root: Path | None = None) -> FastAPI:
    resolved_repo_root = infer_repo_root(repo_root)
    app = FastAPI(
        title="Owli Control UI",
        description="Phase 1 read-only visibility into contracts, datasets, runs, and artifacts.",
        version="0.1.0",
    )
    app.state.repo_root = resolved_repo_root

    def _reader() -> RepositoryReader:
        return RepositoryReader(app.state.repo_root)

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

    return app


app = create_app()
