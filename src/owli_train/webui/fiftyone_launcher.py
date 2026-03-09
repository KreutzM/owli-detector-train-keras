from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a local FiftyOne session for WebUI.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--coco-path", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--status-path", required=True)
    args = parser.parse_args()

    status_path = Path(args.status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import fiftyone as fo

        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=args.images_dir,
            labels_path=args.coco_path,
            name=args.dataset_name,
            overwrite=True,
        )
        session = fo.launch_app(dataset, address="127.0.0.1", port=args.port, auto=False)
        status_path.write_text(
            json.dumps(
                {
                    "status": "ready",
                    "message": "FiftyOne app is ready.",
                    "app_url": f"http://127.0.0.1:{args.port}/",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        session.wait()
        return 0
    except Exception as exc:  # pragma: no cover - exercised through the parent process
        status_path.write_text(
            json.dumps(
                {
                    "status": "error",
                    "message": str(exc),
                    "detail": traceback.format_exc(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
