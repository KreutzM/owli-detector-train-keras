from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from owli_train.webui.models import FiftyOneLaunchResultView, FiftyOneLaunchTargetView


@dataclass
class _ActiveLaunch:
    signature: str
    app_url: str
    process: subprocess.Popen[str]


class FiftyOneService:
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root).resolve()
        self.state_dir = self.repo_root / "work" / "webui" / "fiftyone"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._active: _ActiveLaunch | None = None

    def launch(self, target: FiftyOneLaunchTargetView) -> FiftyOneLaunchResultView:
        if not target.can_launch or target.coco_path is None or target.images_dir is None:
            return FiftyOneLaunchResultView(status="error", message=target.message)
        if importlib.util.find_spec("fiftyone") is None:
            return FiftyOneLaunchResultView(
                status="error",
                message=(
                    "FiftyOne is not installed in this venv. Install it locally with "
                    "`pip install -r requirements/fiftyone.txt`."
                ),
            )

        signature = f"{target.coco_path}|{target.images_dir}"
        with self._lock:
            if self._active is not None and self._active.signature == signature:
                if self._active.process.poll() is None:
                    return FiftyOneLaunchResultView(
                        status="ready",
                        message="Reusing the existing local FiftyOne session.",
                        app_url=self._active.app_url,
                    )
                self._active = None

            self._stop_active_locked()
            return self._start_new_locked(target=target, signature=signature)

    def shutdown(self) -> None:
        with self._lock:
            self._stop_active_locked()

    def _start_new_locked(
        self, *, target: FiftyOneLaunchTargetView, signature: str
    ) -> FiftyOneLaunchResultView:
        assert target.coco_path is not None
        assert target.images_dir is not None

        coco_path = (self.repo_root / target.coco_path).resolve()
        images_dir = (self.repo_root / target.images_dir).resolve()
        if not coco_path.is_file():
            return FiftyOneLaunchResultView(
                status="error",
                message=f"COCO file was not found: {target.coco_path}",
            )
        if not images_dir.is_dir():
            return FiftyOneLaunchResultView(
                status="error",
                message=f"Images directory was not found: {target.images_dir}",
            )

        launch_id = f"{int(time.time() * 1000)}-{os.getpid()}"
        status_path = self.state_dir / f"launch-{launch_id}.json"
        log_path = self.state_dir / f"launch-{launch_id}.log"
        port = self._pick_port()
        app_url = f"http://127.0.0.1:{port}/"
        env = os.environ.copy()
        src_dir = self.repo_root / "src"
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{src_dir}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(src_dir)
        )

        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "owli_train.webui.fiftyone_launcher",
                    "--dataset-name",
                    target.dataset_name or "owli-dataset",
                    "--coco-path",
                    str(coco_path),
                    "--images-dir",
                    str(images_dir),
                    "--port",
                    str(port),
                    "--status-path",
                    str(status_path),
                ],
                cwd=self.repo_root,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )

        ready = self._wait_for_status(
            status_path=status_path, process=process, timeout_seconds=12.0
        )
        if ready.status != "ready":
            self._terminate_process(process)
            return ready

        resolved_app_url = ready.app_url or app_url
        self._active = _ActiveLaunch(
            signature=signature,
            app_url=resolved_app_url,
            process=process,
        )
        return FiftyOneLaunchResultView(
            status="ready",
            message="Started a local FiftyOne session for this dataset.",
            app_url=resolved_app_url,
        )

    def _wait_for_status(
        self,
        *,
        status_path: Path,
        process: subprocess.Popen[str],
        timeout_seconds: float,
    ) -> FiftyOneLaunchResultView:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if status_path.is_file():
                try:
                    payload = json.loads(status_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    break
                if payload.get("status") == "ready":
                    return FiftyOneLaunchResultView(
                        status="ready",
                        message=str(payload.get("message", "FiftyOne is ready.")),
                        app_url=str(payload.get("app_url", "")) or None,
                    )
                return FiftyOneLaunchResultView(
                    status="error",
                    message=str(payload.get("message", "FiftyOne launch failed.")),
                    detail=str(payload.get("detail", "")).strip() or None,
                )
            if process.poll() is not None:
                return FiftyOneLaunchResultView(
                    status="error",
                    message="FiftyOne launcher exited before the app became ready.",
                )
            time.sleep(0.2)
        return FiftyOneLaunchResultView(
            status="error",
            message="Timed out while starting the local FiftyOne session.",
        )

    def _pick_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            return int(sock.getsockname()[1])

    def _stop_active_locked(self) -> None:
        if self._active is None:
            return
        self._terminate_process(self._active.process)
        self._active = None

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
