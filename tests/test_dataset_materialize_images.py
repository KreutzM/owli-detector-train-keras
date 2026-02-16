import json
from pathlib import Path

from typer.testing import CliRunner

from owli_train.cli import app

runner = CliRunner()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _minimal_coco_with_images(file_names: list[str]) -> dict:
    images = [
        {"id": idx + 1, "file_name": name, "width": 64, "height": 64}
        for idx, name in enumerate(file_names)
    ]
    annotations = [
        {
            "id": idx + 1,
            "image_id": idx + 1,
            "category_id": 1,
            "bbox": [1, 1, 10, 10],
            "area": 100,
            "iscrowd": 0,
        }
        for idx, _ in enumerate(file_names)
    ]
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}],
    }


def test_dataset_materialize_images_with_merge_manifest_copy_mode(tmp_path: Path) -> None:
    images_gt = tmp_path / "images_gt"
    images_pseudo = tmp_path / "images_pseudo"
    images_gt.mkdir(parents=True)
    images_pseudo.mkdir(parents=True)
    (images_gt / "a.jpg").write_bytes(b"gt")
    (images_pseudo / "b.jpg").write_bytes(b"ps")

    merged_coco = _minimal_coco_with_images(["gt/a.jpg", "pseudo/b.jpg"])
    merged_path = tmp_path / "merged.json"
    _write_json(merged_path, merged_coco)

    # merge manifest loader requires coco paths to exist.
    (tmp_path / "gt_source.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pseudo_source.json").write_text("{}", encoding="utf-8")

    manifest_path = tmp_path / "merge.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "sources:",
                "  - name: gt",
                "    coco: gt_source.json",
                "    images_dir: images_gt",
                "    file_name_prefix: gt",
                "  - name: pseudo",
                "    coco: pseudo_source.json",
                "    images_dir: images_pseudo",
                "    file_name_prefix: pseudo",
            ]
        ),
        encoding="utf-8",
    )

    out_images = tmp_path / "out_images"
    out_coco = tmp_path / "out_instances.json"

    result = runner.invoke(
        app,
        [
            "dataset",
            "materialize-images",
            "--coco",
            str(merged_path),
            "--merge-manifest",
            str(manifest_path),
            "--out-images-dir",
            str(out_images),
            "--out-coco",
            str(out_coco),
            "--mode",
            "copy",
        ],
    )

    assert result.exit_code == 0
    assert (out_images / "gt" / "a.jpg").is_file()
    assert (out_images / "pseudo" / "b.jpg").is_file()
    assert out_coco.is_file()

    out_payload = json.loads(out_coco.read_text(encoding="utf-8"))
    assert [item["file_name"] for item in out_payload["images"]] == ["gt/a.jpg", "pseudo/b.jpg"]


def test_dataset_materialize_images_with_source_images_dir(tmp_path: Path) -> None:
    source_images = tmp_path / "images"
    source_images.mkdir(parents=True)
    (source_images / "a.jpg").write_bytes(b"a")

    merged_coco = _minimal_coco_with_images(["a.jpg"])
    merged_path = tmp_path / "merged.json"
    _write_json(merged_path, merged_coco)

    out_images = tmp_path / "out_images"
    out_coco = tmp_path / "out_instances.json"

    result = runner.invoke(
        app,
        [
            "dataset",
            "materialize-images",
            "--coco",
            str(merged_path),
            "--source-images-dir",
            str(source_images),
            "--out-images-dir",
            str(out_images),
            "--out-coco",
            str(out_coco),
            "--mode",
            "copy",
        ],
    )

    assert result.exit_code == 0
    assert (out_images / "a.jpg").is_file()
    assert out_coco.is_file()


def test_dataset_materialize_images_rejects_ambiguous_source_images(tmp_path: Path) -> None:
    source_a = tmp_path / "images_a"
    source_b = tmp_path / "images_b"
    source_a.mkdir(parents=True)
    source_b.mkdir(parents=True)
    (source_a / "shared.jpg").write_bytes(b"a")
    (source_b / "shared.jpg").write_bytes(b"b")

    merged_coco = _minimal_coco_with_images(["shared.jpg"])
    merged_path = tmp_path / "merged.json"
    _write_json(merged_path, merged_coco)

    out_images = tmp_path / "out_images"
    out_coco = tmp_path / "out_instances.json"
    result = runner.invoke(
        app,
        [
            "dataset",
            "materialize-images",
            "--coco",
            str(merged_path),
            "--source-images-dir",
            str(source_a),
            "--source-images-dir",
            str(source_b),
            "--out-images-dir",
            str(out_images),
            "--out-coco",
            str(out_coco),
        ],
    )

    assert result.exit_code == 1
    assert "Ambiguous source image" in result.stdout
