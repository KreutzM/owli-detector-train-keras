from owli_train.pseudo_label.coco_writer import (
    build_pseudo_coco,
    build_pseudo_report,
    load_coco80_categories,
    parse_classes_filter,
)


def test_load_coco80_categories_has_expected_entries() -> None:
    categories = load_coco80_categories()
    assert len(categories) == 80
    assert categories[0]["id"] == 1
    assert categories[0]["name"] == "person"


def test_parse_classes_filter_accepts_names_and_ids() -> None:
    categories = load_coco80_categories()
    selected = parse_classes_filter("person,3,5", categories=categories)
    assert selected is not None
    assert 1 in selected
    assert 3 in selected
    assert 5 in selected


def test_build_pseudo_coco_schema_contains_score_field() -> None:
    categories = load_coco80_categories()
    images = [{"id": 1, "file_name": "img/a.jpg", "width": 640, "height": 480}]
    detections = [
        {"image_id": 1, "category_id": 1, "bbox": [10.0, 20.0, 30.0, 40.0], "score": 0.92}
    ]
    coco = build_pseudo_coco(images=images, detections=detections, categories=categories)
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 1
    ann = coco["annotations"][0]
    assert ann["image_id"] == 1
    assert ann["category_id"] == 1
    assert ann["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert ann["score"] == 0.92


def test_build_pseudo_report_aggregates_counts_and_runtime() -> None:
    categories = load_coco80_categories()
    detections = [
        {"image_id": 1, "category_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.9},
        {"image_id": 2, "category_id": 3, "bbox": [2.0, 2.0, 8.0, 8.0], "score": 0.7},
        {"image_id": 2, "category_id": 3, "bbox": [4.0, 4.0, 5.0, 5.0], "score": 0.6},
    ]
    report = build_pseudo_report(
        num_images=2,
        detections=detections,
        categories=categories,
        total_seconds=1.0,
        teacher_source="tfhub://dummy",
        batch_size=4,
        input_size=640,
        score_threshold=0.6,
        max_detections_per_image=50,
    )

    assert report["num_images"] == 2
    assert report["total_detections"] == 3
    assert report["average_detections_per_image"] == 1.5
    assert report["runtime"]["images_per_second"] == 2.0
    assert report["per_class_counts"]["person"] == 1
    assert report["per_class_counts"]["car"] == 2
    assert len(report["score_histogram"]) == 10
