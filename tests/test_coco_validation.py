from pathlib import Path

import pytest

from owli_train.data.coco import validate_coco


def _minimal_coco() -> dict:
    return {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "person"}],
    }


def test_validate_enforces_required_keys():
    obj = _minimal_coco()
    del obj["categories"]

    with pytest.raises(ValueError, match="Missing COCO key: categories"):
        validate_coco(obj)


def test_validate_enforces_unique_ids():
    obj = _minimal_coco()
    obj["images"] = [{"id": 1, "file_name": "a.jpg"}, {"id": 1, "file_name": "b.jpg"}]

    with pytest.raises(ValueError, match="images contains duplicate id"):
        validate_coco(obj)


def test_validate_enforces_category_and_image_refs():
    obj = _minimal_coco()
    obj["annotations"][0]["image_id"] = 999

    with pytest.raises(ValueError, match="unknown image_id"):
        validate_coco(obj)

    obj = _minimal_coco()
    obj["annotations"][0]["category_id"] = 999

    with pytest.raises(ValueError, match="unknown category_id"):
        validate_coco(obj)


def test_validate_enforces_bbox_size():
    obj = _minimal_coco()
    obj["annotations"][0]["bbox"] = [0, 0, 0, 10]

    with pytest.raises(ValueError, match="width and height must be > 0"):
        validate_coco(obj)


def test_validate_images_dir_checks_files_exist(tmp_path: Path):
    obj = _minimal_coco()
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "a.jpg").write_bytes(b"not-a-real-jpeg")

    summary = validate_coco(obj, images_dir=images_dir)
    assert summary.images == 1

    missing = _minimal_coco()
    missing["images"][0]["file_name"] = "missing.jpg"

    with pytest.raises(ValueError, match="does not exist"):
        validate_coco(missing, images_dir=images_dir)
