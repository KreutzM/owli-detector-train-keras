from __future__ import annotations

import argparse
from pathlib import Path

from owli_train.webui.jobs import run_job_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single persisted WebUI job.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--job-file", required=True)
    args = parser.parse_args()

    result = run_job_file(
        repo_root=Path(args.repo_root),
        job_file=Path(args.job_file),
    )
    return 0 if result.status == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
