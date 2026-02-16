import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from owli_train.cli import app
from owli_train.data.merge_coco import CocoMergeError, merge_coco_from_manifest

runner = CliRunner()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_merge_coco_from_manifest_merges_gt_and_pseudo(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "a.jpg").write_bytes(b"x")
    (images_dir / "b.jpg").write_bytes(b"x")

    gt_coco = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "b.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [60, 60, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "human"},
            {"id": 2, "name": "helmet"},
        ],
    }
    pseudo_coco = {
        "images": [
            {"id": 10, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 20, "file_name": "b.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 10,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "score": 0.95,
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 11,
                "image_id": 20,
                "category_id": 2,
                "bbox": [5, 5, 20, 20],
                "score": 0.94,
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 12,
                "image_id": 20,
                "category_id": 1,
                "bbox": [20, 20, 15, 15],
                "score": 0.30,
                "area": 225,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "bicycle"},
        ],
    }

    gt_path = tmp_path / "gt.json"
    pseudo_path = tmp_path / "pseudo.json"
    _write_json(gt_path, gt_coco)
    _write_json(pseudo_path, pseudo_coco)

    label_map_path = tmp_path / "label_map.yaml"
    label_map_path.write_text("map:\n  human: person\n", encoding="utf-8")

    manifest_path = tmp_path / "merge.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "sources:",
                "  - name: gt",
                "    coco: gt.json",
                "    images_dir: images",
                "    label_map: label_map.yaml",
                "  - name: pseudo",
                "    coco: pseudo.json",
                "    images_dir: images",
                "    pseudo: true",
                "    score_threshold: 0.6",
                "settings:",
                "  same_class_iou: 0.75",
                "  pseudo_block_iou: 0.6",
            ]
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "merged.json"
    artifacts = merge_coco_from_manifest(manifest_path=manifest_path, out_path=out_path)

    assert artifacts.coco_path == out_path
    assert artifacts.images == 2
    assert artifacts.annotations == 3
    assert artifacts.categories == 3

    merged = json.loads(out_path.read_text(encoding="utf-8"))
    assert [item["name"] for item in merged["categories"]] == ["helmet", "person", "bicycle"]

    image_b_by_id = {
        int(image["id"]): image for image in merged["images"] if image["file_name"] == "b.jpg"
    }
    assert len(image_b_by_id) == 1
    b_image_id = next(iter(image_b_by_id))

    category_name_by_id = {int(cat["id"]): str(cat["name"]) for cat in merged["categories"]}
    ann_names_for_b = [
        category_name_by_id[int(ann["category_id"])]
        for ann in merged["annotations"]
        if int(ann["image_id"]) == b_image_id
    ]
    assert ann_names_for_b == ["bicycle"]

    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert report["drops"]["pseudo_overlap_gt"] == 1
    assert report["drops"]["pseudo_low_score"] == 1


def test_merge_coco_from_manifest_rejects_duplicate_file_names_without_prefix(
    tmp_path: Path,
) -> None:
    images_a = tmp_path / "images_a"
    images_b = tmp_path / "images_b"
    images_a.mkdir(parents=True)
    images_b.mkdir(parents=True)
    (images_a / "shared.jpg").write_bytes(b"a")
    (images_b / "shared.jpg").write_bytes(b"b")

    coco_a = {
        "images": [{"id": 1, "file_name": "shared.jpg", "width": 32, "height": 32}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 1, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    coco_b = {
        "images": [{"id": 1, "file_name": "shared.jpg", "width": 32, "height": 32}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [2, 2, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "car"}],
    }

    _write_json(tmp_path / "a.json", coco_a)
    _write_json(tmp_path / "b.json", coco_b)

    manifest_path = tmp_path / "merge.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "sources:",
                "  - name: a",
                "    coco: a.json",
                "    images_dir: images_a",
                "  - name: b",
                "    coco: b.json",
                "    images_dir: images_b",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(CocoMergeError, match="duplicate file_name"):
        merge_coco_from_manifest(
            manifest_path=manifest_path,
            out_path=tmp_path / "merged.json",
        )


def test_dataset_merge_coco_cli(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "a.jpg").write_bytes(b"x")

    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [5, 5, 20, 20],
                "area": 400,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    _write_json(tmp_path / "a.json", coco)

    manifest_path = tmp_path / "merge.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "sources:",
                "  - name: a",
                "    coco: a.json",
                "    images_dir: images",
            ]
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "merged.json"
    report_path = tmp_path / "merged.report.json"
    result = runner.invoke(
        app,
        [
            "dataset",
            "merge",
            "coco",
            "--manifest",
            str(manifest_path),
            "--out",
            str(out_path),
            "--report-out",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert out_path.is_file()
    assert report_path.is_file()
