from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from owli_train.webui.models import (
    ArtifactDirectoryView,
    CompareClassScopeOptionView,
    ComparePerClassCellView,
    ComparePerClassRowView,
    CompareRowView,
    CompareRunOptionView,
    CompareTargetOptionView,
    ConfigGroupView,
    ConfigReferenceView,
    ContractClassView,
    ContractRoleView,
    ContractView,
    CountRowView,
    DatasetDetailView,
    EvalDetailView,
    FiftyOneLaunchTargetView,
    GoldenDetailView,
    GoldenDetectionView,
    LabelValueView,
    LinkedArtifactView,
    MetricRowView,
    PerClassMetricRowView,
    RepoPathView,
    RepositoryViewModel,
    RunDetailView,
    RunsCompareView,
)

CURATED_DOCS: tuple[tuple[str, str, str], ...] = (
    ("README", "README.md", "Project overview and quickstart."),
    ("Runbook", "docs/runbook.md", "Dataset and artifact workflow reference."),
    ("Training plan", "docs/MVP_Training_Plan.md", "Current BA-v1 to BA-v2 transition plan."),
    ("BA-v1 labelset", "docs/BA_v1_Labelset.md", "Historical verified interim ontology."),
    (
        "BA-v2 hazard labelset",
        "docs/BA_v2_Hazard_Labelset.md",
        "Preferred hazard-centered ontology.",
    ),
    (
        "Latest task report",
        "docs/reviews/Codex-Task-Report_last.md",
        "Most recent Codex task execution report.",
    ),
)

CURATED_ARTIFACT_ROOTS: tuple[tuple[str, str, str], ...] = (
    ("Datasets root", "work/datasets", "Prepared dataset outputs and intermediate artifacts."),
    ("Runs root", "work/runs", "Training and export run directories."),
    ("WebUI jobs root", "work/webui/jobs", "Persistent Phase-2 WebUI job records and logs."),
    ("Splits root", "work/splits", "Deterministic split artifacts."),
    ("Reports root", "work/reports", "Evaluation and inspection reports."),
    ("Outputs root", "outputs", "Model outputs configured by training backends."),
    ("Data root", "data", "Raw or imported source data."),
)

CONTRACT_ORDER: tuple[str, ...] = ("ba_v1", "ba_v2_hazard")
COMPARE_METRIC_KEYS: tuple[str, ...] = ("AP", "AP50", "AP75", "AR100", "precision", "recall")
COMPARE_PER_CLASS_METRIC_KEYS: tuple[str, ...] = ("precision", "recall", "tp", "fp", "fn")
COMPARE_CLASS_SCOPE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("ba_core", "BA core only"),
    ("ba_core_rehearsal", "BA core + rehearsal"),
)
CURATED_COMPARE_CLASS_GROUPS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("ba_core", "obstacle_bump", "BA core", ("obstacle_bump",)),
    (
        "ba_core",
        "obstacle_fence / obstacle_fence_rail",
        "BA core",
        ("obstacle_fence", "obstacle_fence_rail"),
    ),
    (
        "ba_core",
        "obstacle_hole / obstacle_hole_dropoff",
        "BA core",
        ("obstacle_hole", "obstacle_hole_dropoff"),
    ),
    ("ba_core", "obstacle_pole", "BA core", ("obstacle_pole",)),
    ("ba_core", "obstacle_ground", "BA core", ("obstacle_ground",)),
    ("ba_core", "obstacle_barrier", "BA core", ("obstacle_barrier",)),
    ("ba_core_rehearsal", "person", "Rehearsal", ("person",)),
    ("ba_core_rehearsal", "bicycle", "Rehearsal", ("bicycle",)),
    ("ba_core_rehearsal", "motorcycle", "Rehearsal", ("motorcycle",)),
    ("ba_core_rehearsal", "car", "Rehearsal", ("car",)),
    ("ba_core_rehearsal", "bus", "Rehearsal", ("bus",)),
    ("ba_core_rehearsal", "truck", "Rehearsal", ("truck",)),
)
CURATED_COMPARE_LABELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Stage-3 baseline",
        (
            "ba-mvp-stage3-",
            "ba_mvp_stage3",
            "stage3_balanced_multisource",
        ),
    ),
    (
        "Stage-4 replay baseline",
        (
            "ba-mvp-stage4-",
            "ba_mvp_stage4",
            "stage4_with_coco_replay",
            "coco_replay",
        ),
    ),
    (
        "Stage-3-plus-crops baseline",
        (
            "ba-mvp-stage3-plus-crops",
            "ba_mvp_stage3_plus_crops",
            "stage3-plus-crops",
            "stage3_plus_crops",
        ),
    ),
    (
        "BA-v2 hazard",
        (
            "ba-v2-hazard",
            "ba_v2_hazard",
        ),
    ),
)


@dataclass(frozen=True)
class _CompareCandidate:
    run_id: str
    run_relative_path: str
    run_display_name: str
    curated_label: str | None
    stage_label: str | None
    dataset_label: str
    target_key: str
    target_label: str
    config_path: str | None
    config_note: str | None
    eval_relative_path: str
    eval_label: str
    eval_sort_value: str
    golden_relative_path: str | None
    metrics: dict[str, str]
    per_class: dict[str, dict[str, Any]]


def infer_repo_root(start: Path | None = None) -> Path:
    if start is not None:
        return Path(start).resolve()

    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "README.md").is_file():
            return candidate
    return current.parents[3]


class RepositoryReader:
    def __init__(self, repo_root: Path | None = None):
        self.repo_root = infer_repo_root(repo_root)

    def build_view_model(self) -> RepositoryViewModel:
        return RepositoryViewModel(
            repo_root=str(self.repo_root),
            docs=self.load_docs(),
            contracts=self.load_contracts(),
            artifact_roots=self.load_artifact_roots(),
            recent_datasets=self.load_recent_datasets(),
            recent_runs=self.load_recent_runs(),
            config_groups=self.load_config_groups(),
        )

    def load_docs(self) -> list[RepoPathView]:
        return [
            self._build_repo_path(label, relative_path, note)
            for label, relative_path, note in CURATED_DOCS
        ]

    def load_contracts(self) -> list[ContractView]:
        contract_dir = self.repo_root / "configs" / "label_contracts"
        ordered_paths: list[Path] = []
        for key in CONTRACT_ORDER:
            path = contract_dir / f"{key}.yaml"
            if path.is_file():
                ordered_paths.append(path)
        for path in sorted(contract_dir.glob("*.yaml")):
            if path not in ordered_paths:
                ordered_paths.append(path)

        contracts: list[ContractView] = []
        for path in ordered_paths:
            payload = self._load_yaml(path)
            class_names = self._as_list_of_strings(payload.get("class_names"))
            role_views = [
                ContractRoleView(
                    name=str(role_name), class_names=self._as_list_of_strings(role_values)
                )
                for role_name, role_values in self._as_mapping(payload.get("roles")).items()
            ]
            classes_payload = payload.get("classes")
            class_rows: list[ContractClassView] = []
            if isinstance(classes_payload, list):
                for index, item in enumerate(classes_payload, start=1):
                    if not isinstance(item, dict):
                        continue
                    class_rows.append(
                        ContractClassView(
                            order=index,
                            name=str(item.get("name", "")).strip(),
                            role=str(item.get("role", "")).strip(),
                            rationale=self._optional_text(item.get("rationale")),
                        )
                    )
            if not class_rows:
                class_rows = [
                    ContractClassView(order=index, name=name, role="", rationale=None)
                    for index, name in enumerate(class_names, start=1)
                ]

            relative_path = self._relative_path(path)
            contracts.append(
                ContractView(
                    key=path.stem,
                    title=str(payload.get("version") or path.stem),
                    relative_path=relative_path,
                    exists=path.is_file(),
                    status=self._optional_text(payload.get("status")),
                    purpose=self._optional_text(payload.get("purpose")),
                    class_count=len(class_names),
                    roles=role_views,
                    classes=class_rows,
                    out_of_scope=self._as_list_of_strings(payload.get("out_of_scope")),
                    modified_at=self._format_mtime(path),
                )
            )
        return contracts

    def load_artifact_roots(self) -> list[RepoPathView]:
        return [
            self._build_repo_path(label, relative_path, note)
            for label, relative_path, note in CURATED_ARTIFACT_ROOTS
        ]

    def load_recent_datasets(self, limit: int = 8) -> list[ArtifactDirectoryView]:
        root = self.repo_root / "work" / "datasets"
        return self._scan_artifact_directories(
            root=root,
            limit=limit,
            highlight_patterns=(
                "instances*.json",
                "*.csv",
                "class_names.json",
                "qc_report.json",
                "splits.json",
            ),
        )

    def load_recent_runs(self, limit: int = 8) -> list[ArtifactDirectoryView]:
        root = self.repo_root / "work" / "runs"
        return self._scan_artifact_directories(
            root=root,
            limit=limit,
            highlight_patterns=(
                "artifacts/*.tflite",
                "artifacts/labels.txt",
                "reports/*.json",
                "reports/*.md",
                "config.yaml",
                "config_snapshot.yaml",
                "mapping_files.json",
                "mapping_snapshot.json",
            ),
        )

    def load_config_groups(self) -> list[ConfigGroupView]:
        return [
            ConfigGroupView(
                title="Dataset prep configs",
                description="Configs that point to stable dataset outputs or source inputs.",
                items=self._load_dataset_config_references(),
            ),
            ConfigGroupView(
                title="Merge manifests",
                description="Curated merge manifests that assemble multi-source COCO inputs.",
                items=self._load_merge_config_references(),
            ),
            ConfigGroupView(
                title="Training configs",
                description="Model Maker training configs and their referenced inputs.",
                items=self._load_training_config_references(),
            ),
        ]

    def load_dataset_detail(self, relative_path: str) -> DatasetDetailView | None:
        dataset_dir = self._resolve_allowed_path(
            relative_path,
            required_kind="dir",
            allowed_roots=(
                "work/datasets",
                "work/splits",
                "work/webui/materialized",
                "work/webui/splits",
            ),
        )
        if dataset_dir is None:
            return None

        primary_coco = self._find_primary_dataset_coco(dataset_dir)
        coco_payload = self._load_json(primary_coco) if primary_coco is not None else {}
        images = coco_payload.get("images", []) if isinstance(coco_payload, dict) else []
        annotations = coco_payload.get("annotations", []) if isinstance(coco_payload, dict) else []
        categories = coco_payload.get("categories", []) if isinstance(coco_payload, dict) else []

        category_names = {
            int(item.get("id")): str(item.get("name", "")).strip()
            for item in categories
            if isinstance(item, dict) and item.get("id") is not None
        }
        annotation_counts: Counter[str] = Counter()
        for item in annotations:
            if not isinstance(item, dict):
                continue
            category_id = item.get("category_id")
            if category_id is None:
                continue
            category_name = category_names.get(int(category_id), f"id:{category_id}")
            annotation_counts[category_name] += 1

        split_counts = self._load_split_counts(dataset_dir)
        qc_summary = self._load_qc_summary(dataset_dir)
        images_dir = dataset_dir / "images"
        related_files = [
            self._path_view_from_absolute(dataset_dir, label="dataset directory"),
        ]
        if primary_coco is not None:
            related_files.append(self._path_view_from_absolute(primary_coco, label="primary COCO"))
        if images_dir.is_dir():
            related_files.append(self._path_view_from_absolute(images_dir, label="images dir"))

        qc_path = dataset_dir / "qc_report.json"
        if qc_path.is_file():
            related_files.append(self._path_view_from_absolute(qc_path, label="QC report"))

        split_path = self._find_split_file(dataset_dir)
        if split_path is not None:
            related_files.append(self._path_view_from_absolute(split_path, label="splits"))

        summary = [
            LabelValueView(label="Dataset path", value=self._relative_path(dataset_dir)),
            LabelValueView(label="Images", value=str(len(images))),
            LabelValueView(label="Annotations", value=str(len(annotations))),
            LabelValueView(label="Categories", value=str(len(categories))),
        ]
        if primary_coco is not None:
            summary.append(
                LabelValueView(label="Primary COCO", value=self._relative_path(primary_coco))
            )
        if images_dir.is_dir():
            summary.append(
                LabelValueView(label="Images dir", value=self._relative_path(images_dir))
            )

        return DatasetDetailView(
            title=dataset_dir.name,
            relative_path=self._relative_path(dataset_dir),
            primary_coco_path=self._relative_path(primary_coco)
            if primary_coco is not None
            else None,
            images_dir=self._relative_path(images_dir) if images_dir.is_dir() else None,
            fiftyone_target=self._build_dataset_fiftyone_target(
                dataset_dir=dataset_dir,
                primary_coco=primary_coco,
                images_dir=images_dir,
            ),
            summary=summary,
            class_distribution=[
                CountRowView(label=label, value=value)
                for label, value in sorted(
                    annotation_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
            split_counts=split_counts,
            qc_summary=qc_summary,
            related_files=related_files,
            related_configs=self._find_config_references_for_prefix(
                self._relative_path(dataset_dir)
            ),
        )

    def load_run_detail(self, relative_path: str) -> RunDetailView | None:
        run_dir = self._resolve_allowed_path(
            relative_path,
            required_kind="dir",
            allowed_roots=("work/runs",),
        )
        if run_dir is None:
            return None

        artifact_files = self._list_known_files(run_dir / "artifacts")
        report_files = self._list_known_files(run_dir / "reports")
        config_candidates = [
            run_dir / "config.yaml",
            run_dir / "config_snapshot.yaml",
            run_dir / "mapping_files.json",
            run_dir / "mapping_snapshot.json",
            run_dir / "class_names.json",
        ]
        config_files = [
            self._path_view_from_absolute(path, label=path.name)
            for path in config_candidates
            if path.exists()
        ]

        eval_reports = [
            self._linked_report_view(path, note=self._eval_headline(path))
            for path in sorted((run_dir / "reports").glob("eval*.json"))
        ]
        golden_reports = [
            self._linked_report_view(path, note=self._golden_headline(path))
            for path in sorted((run_dir / "reports").glob("golden*.json"))
        ]

        summary = [
            LabelValueView(label="Run id", value=run_dir.name),
            LabelValueView(label="Run path", value=self._relative_path(run_dir)),
            LabelValueView(label="Artifacts", value=str(len(artifact_files))),
            LabelValueView(label="Reports", value=str(len(report_files))),
        ]
        return RunDetailView(
            run_id=run_dir.name,
            relative_path=self._relative_path(run_dir),
            summary=summary,
            config_files=config_files,
            artifact_files=artifact_files,
            report_files=report_files,
            eval_reports=eval_reports,
            golden_reports=golden_reports,
        )

    def load_runs_compare(
        self,
        *,
        selected_run_paths: list[str] | None = None,
        target_key: str | None = None,
        class_scope_key: str | None = None,
    ) -> RunsCompareView:
        candidates = self._load_compare_candidates()
        selected_paths = {path for path in (selected_run_paths or []) if path.strip()}
        run_paths = {candidate.run_relative_path for candidate in candidates}
        filtered_candidates = [
            candidate
            for candidate in candidates
            if not selected_paths or candidate.run_relative_path in selected_paths
        ]

        run_options = self._build_compare_run_options(
            all_run_paths=run_paths,
            candidates=candidates,
            selected_paths=selected_paths,
        )
        target_options = self._build_compare_target_options(
            candidates=filtered_candidates,
            requested_target_key=target_key,
        )
        selected_target = next(
            (item for item in target_options if item.selected),
            None,
        )
        rows = [
            self._candidate_to_compare_row(candidate)
            for candidate in filtered_candidates
            if selected_target is not None and candidate.target_key == selected_target.key
        ]
        selected_candidates = [
            candidate
            for candidate in filtered_candidates
            if selected_target is not None and candidate.target_key == selected_target.key
        ]
        class_scope_options = self._build_compare_class_scope_options(
            requested_scope_key=class_scope_key
        )
        selected_class_scope = next(
            (item for item in class_scope_options if item.selected),
            class_scope_options[0],
        )
        per_class_rows = self._build_compare_per_class_rows(
            candidates=selected_candidates,
            scope_key=selected_class_scope.key,
        )

        if selected_target is None:
            selection_summary = "No comparable eval JSON reports were found under work/runs."
        elif rows:
            selection_summary = (
                f"Showing {len(rows)} eval rows across {len({row.run_relative_path for row in rows})} "
                f"runs for {selected_target.label}."
            )
        elif selected_paths and selected_paths.isdisjoint(run_paths):
            selection_summary = "The selected run paths were not found under work/runs."
        else:
            selection_summary = "No eval rows match the current run and target selection."

        scope_note = (
            "This page compares structured eval JSON reports only. It intentionally stays small: "
            "global metrics, dataset/stage context, and links back to run, eval, and golden details."
        )
        per_class_note = (
            "Per-class rows use a small curated class list and only exact class names or the listed "
            "historical aliases inside each row. Missing values stay visible as '-'."
        )
        return RunsCompareView(
            selected_target_key=selected_target.key if selected_target is not None else None,
            selected_target_label=selected_target.label if selected_target is not None else None,
            target_options=target_options,
            run_options=run_options,
            class_scope_key=selected_class_scope.key,
            class_scope_options=class_scope_options,
            metric_keys=list(COMPARE_METRIC_KEYS),
            rows=rows,
            per_class_metric_keys=list(COMPARE_PER_CLASS_METRIC_KEYS),
            per_class_rows=per_class_rows,
            per_class_note=per_class_note,
            selection_summary=selection_summary,
            scope_note=scope_note,
        )

    def load_eval_detail(self, relative_path: str) -> EvalDetailView | None:
        report_path = self._resolve_allowed_path(
            relative_path,
            required_kind="file",
            allowed_roots=("work",),
        )
        if report_path is None:
            return None

        if report_path.suffix.lower() == ".md":
            return EvalDetailView(
                title=report_path.name,
                relative_path=self._relative_path(report_path),
                summary=[
                    LabelValueView(label="Report path", value=self._relative_path(report_path)),
                    LabelValueView(label="Format", value="markdown"),
                ],
                fiftyone_target=self._build_eval_fiftyone_target(
                    report_path=report_path,
                    payload={},
                ),
                metrics=[],
                summary_counts=[],
                per_class_headers=[],
                per_class_rows=[],
                related_run_path=self._infer_run_path(report_path),
                sibling_reports=self._load_sibling_eval_reports(report_path),
                raw_excerpt=self._read_excerpt(report_path),
            )

        payload = self._load_json(report_path)
        if not isinstance(payload, dict):
            return None

        summary = self._label_values_from_mapping(
            payload,
            keys=(
                "created_at",
                "run_dir",
                "model_path",
                "coco_path",
                "images_dir",
                "bbox_format",
                "score_threshold",
                "map_score_threshold",
                "max_detections_per_image",
                "limit_images",
                "num_threads",
                "num_workers",
                "num_eval_images",
                "num_detections",
            ),
        )
        summary.insert(
            0, LabelValueView(label="Report path", value=self._relative_path(report_path))
        )

        metrics = self._metric_rows(payload.get("metrics"))
        summary_counts = self._metric_rows(payload.get("summary_counts"))
        per_class_headers, per_class_rows = self._per_class_rows(payload.get("per_class"))
        return EvalDetailView(
            title=report_path.name,
            relative_path=self._relative_path(report_path),
            summary=summary,
            fiftyone_target=self._build_eval_fiftyone_target(
                report_path=report_path,
                payload=payload,
            ),
            metrics=metrics,
            summary_counts=summary_counts,
            per_class_headers=per_class_headers,
            per_class_rows=per_class_rows,
            related_run_path=self._infer_run_path(report_path),
            sibling_reports=self._load_sibling_eval_reports(report_path),
            raw_excerpt=None,
        )

    def load_golden_detail(self, relative_path: str) -> GoldenDetailView | None:
        report_path = self._resolve_allowed_path(
            relative_path,
            required_kind="file",
            allowed_roots=("work",),
        )
        if report_path is None:
            return None

        payload = self._load_json(report_path)
        if not isinstance(payload, dict):
            return None

        detections_payload = payload.get("detections", [])
        detections: list[GoldenDetectionView] = []
        if isinstance(detections_payload, list):
            for item in detections_payload:
                if not isinstance(item, dict):
                    continue
                bbox = item.get("bbox")
                detections.append(
                    GoldenDetectionView(
                        class_name=str(item.get("class_name", "")).strip() or "-",
                        score=self._format_number(item.get("score")),
                        bbox=self._format_bbox(bbox),
                    )
                )

        summary = [
            LabelValueView(label="Report path", value=self._relative_path(report_path)),
            LabelValueView(label="Created", value=str(payload.get("created_at", "-"))),
            LabelValueView(label="Model path", value=str(payload.get("model_path", "-"))),
            LabelValueView(label="Image path", value=str(payload.get("image_path", "-"))),
            LabelValueView(label="Detections", value=str(len(detections))),
        ]
        contract = self._label_values_from_mapping(
            payload.get("contract"),
            keys=(
                "class_labels_source",
                "score_threshold",
                "max_results",
                "bbox_format",
                "coordinates",
            ),
        )
        model_metadata = self._label_values_from_mapping(payload.get("model_metadata"))
        inspect_tflite = self._label_values_from_mapping(payload.get("inspect_tflite"))
        return GoldenDetailView(
            title=report_path.name,
            relative_path=self._relative_path(report_path),
            summary=summary,
            contract=contract,
            model_metadata=model_metadata,
            inspect_tflite=inspect_tflite,
            detections=detections,
            related_run_path=self._infer_run_path(report_path),
        )

    def _load_compare_candidates(self) -> list[_CompareCandidate]:
        runs_root = self.repo_root / "work" / "runs"
        if not runs_root.is_dir():
            return []

        candidates: list[_CompareCandidate] = []
        for run_dir in sorted(
            (path for path in runs_root.iterdir() if path.is_dir()), reverse=True
        ):
            run_relative_path = self._relative_path(run_dir)
            config_path, config_note = self._resolve_run_config_reference(run_dir)
            golden_relative_path = self._find_primary_golden_report(run_dir)
            curated_label = self._infer_curated_compare_label(
                run_dir.name,
                run_relative_path,
                config_path,
            )
            for eval_path in sorted((run_dir / "reports").glob("eval*.json")):
                payload = self._load_json(eval_path)
                if not payload:
                    continue
                target_key, target_label, dataset_label, stage_label = (
                    self._describe_compare_target(
                        eval_path=eval_path,
                        payload=payload,
                    )
                )
                candidates.append(
                    _CompareCandidate(
                        run_id=run_dir.name,
                        run_relative_path=run_relative_path,
                        run_display_name=curated_label or run_dir.name,
                        curated_label=curated_label,
                        stage_label=stage_label or curated_label,
                        dataset_label=dataset_label,
                        target_key=target_key,
                        target_label=target_label,
                        config_path=config_path,
                        config_note=config_note,
                        eval_relative_path=self._relative_path(eval_path),
                        eval_label=eval_path.name,
                        eval_sort_value=self._optional_text(payload.get("created_at"))
                        or self._format_mtime(eval_path)
                        or eval_path.name,
                        golden_relative_path=golden_relative_path,
                        metrics=self._extract_compare_metrics(payload),
                        per_class=self._extract_compare_per_class(payload),
                    )
                )
        candidates.sort(key=self._compare_candidate_sort_key)
        return candidates

    def _build_compare_run_options(
        self,
        *,
        all_run_paths: set[str],
        candidates: list[_CompareCandidate],
        selected_paths: set[str],
    ) -> list[CompareRunOptionView]:
        run_details: dict[str, tuple[str, str | None, int]] = {}
        for candidate in candidates:
            current = run_details.get(candidate.run_relative_path)
            count = 1 if current is None else current[2] + 1
            run_details[candidate.run_relative_path] = (
                candidate.run_id,
                candidate.curated_label,
                count,
            )
        return [
            CompareRunOptionView(
                run_id=run_details[path][0],
                relative_path=path,
                selected=not selected_paths or path in selected_paths,
                eval_count=run_details[path][2],
                curated_label=run_details[path][1],
            )
            for path in sorted(all_run_paths, reverse=True)
            if path in run_details
        ]

    def _build_compare_target_options(
        self,
        *,
        candidates: list[_CompareCandidate],
        requested_target_key: str | None,
    ) -> list[CompareTargetOptionView]:
        grouped: dict[str, tuple[str, int]] = {}
        for candidate in candidates:
            label, count = grouped.get(candidate.target_key, (candidate.target_label, 0))
            grouped[candidate.target_key] = (label, count + 1)

        ordered = sorted(grouped.items(), key=lambda item: (-item[1][1], item[1][0].lower()))
        default_target_key = requested_target_key if requested_target_key in grouped else None
        if default_target_key is None and ordered:
            default_target_key = ordered[0][0]

        return [
            CompareTargetOptionView(
                key=key,
                label=label,
                count=count,
                selected=key == default_target_key,
            )
            for key, (label, count) in ordered
        ]

    def _build_compare_class_scope_options(
        self,
        *,
        requested_scope_key: str | None,
    ) -> list[CompareClassScopeOptionView]:
        default_scope_key = requested_scope_key
        valid_keys = {key for key, _label in COMPARE_CLASS_SCOPE_OPTIONS}
        if default_scope_key not in valid_keys:
            default_scope_key = "ba_core"
        return [
            CompareClassScopeOptionView(
                key=key,
                label=label,
                selected=key == default_scope_key,
            )
            for key, label in COMPARE_CLASS_SCOPE_OPTIONS
        ]

    def _build_compare_per_class_rows(
        self,
        *,
        candidates: list[_CompareCandidate],
        scope_key: str,
    ) -> list[ComparePerClassRowView]:
        rows: list[ComparePerClassRowView] = []
        include_rehearsal = scope_key == "ba_core_rehearsal"
        for row_scope, label, role, aliases in CURATED_COMPARE_CLASS_GROUPS:
            if row_scope == "ba_core_rehearsal" and not include_rehearsal:
                continue
            cells = [
                self._build_compare_per_class_cell(candidate, aliases=aliases)
                for candidate in candidates
            ]
            if not any(cell.matched_class_name is not None for cell in cells):
                continue
            note = None
            if len(aliases) > 1:
                note = "Matches " + " or ".join(aliases) + "."
            rows.append(
                ComparePerClassRowView(
                    label=label,
                    role=role,
                    note=note,
                    cells=cells,
                )
            )
        return rows

    def _build_compare_per_class_cell(
        self,
        candidate: _CompareCandidate,
        *,
        aliases: tuple[str, ...],
    ) -> ComparePerClassCellView:
        for alias in aliases:
            payload = candidate.per_class.get(alias.lower())
            if payload is None:
                continue
            return ComparePerClassCellView(
                matched_class_name=str(payload.get("__matched_name__", alias)),
                metrics={
                    key: self._format_number(payload.get(key))
                    for key in COMPARE_PER_CLASS_METRIC_KEYS
                },
            )
        return ComparePerClassCellView(
            matched_class_name=None,
            metrics={key: "-" for key in COMPARE_PER_CLASS_METRIC_KEYS},
        )

    def _candidate_to_compare_row(self, candidate: _CompareCandidate) -> CompareRowView:
        return CompareRowView(
            run_id=candidate.run_id,
            run_relative_path=candidate.run_relative_path,
            run_display_name=candidate.run_display_name,
            curated_label=candidate.curated_label,
            stage_label=candidate.stage_label,
            dataset_label=candidate.dataset_label,
            config_path=candidate.config_path,
            config_note=candidate.config_note,
            eval_relative_path=candidate.eval_relative_path,
            eval_label=candidate.eval_label,
            golden_relative_path=candidate.golden_relative_path,
            metrics=candidate.metrics,
        )

    def _compare_candidate_sort_key(self, candidate: _CompareCandidate) -> tuple[object, ...]:
        curated_order = {
            label: index for index, (label, _tokens) in enumerate(CURATED_COMPARE_LABELS, start=1)
        }
        return (
            candidate.target_label.lower(),
            curated_order.get(candidate.curated_label or "", 999),
            candidate.run_id.lower(),
            candidate.eval_sort_value,
            candidate.eval_label.lower(),
        )

    def _describe_compare_target(
        self,
        *,
        eval_path: Path,
        payload: dict[str, Any],
    ) -> tuple[str, str, str, str | None]:
        coco_value = self._optional_text(payload.get("coco_path"))
        resolved_coco = self._resolve_repo_recorded_path(coco_value)
        if resolved_coco is not None:
            relative = self._relative_path(resolved_coco)
            parts = resolved_coco.parts
            if len(parts) >= 4 and parts[-4:-2] == ("work", "splits"):
                dataset_name = parts[-2]
                split_name = self._infer_split_name_from_path(resolved_coco)
                label = f"{dataset_name} / {split_name.upper()}"
                return (
                    f"split:{dataset_name}:{split_name}",
                    label,
                    dataset_name,
                    self._infer_curated_compare_label(relative, dataset_name),
                )
            if len(parts) >= 4 and parts[-4:-2] == ("work", "datasets"):
                dataset_name = parts[-2]
                label = dataset_name
                return (
                    f"dataset:{dataset_name}:{resolved_coco.name}",
                    label,
                    dataset_name,
                    self._infer_curated_compare_label(relative, dataset_name),
                )
            return (
                f"path:{relative}",
                relative,
                relative,
                self._infer_curated_compare_label(relative),
            )

        fallback = eval_path.stem.replace("_", " ")
        return (
            f"report:{eval_path.name}",
            fallback,
            fallback,
            self._infer_curated_compare_label(fallback),
        )

    def _extract_compare_metrics(self, payload: dict[str, Any]) -> dict[str, str]:
        metrics = payload.get("metrics")
        counts = payload.get("summary_counts")
        metric_map = metrics if isinstance(metrics, dict) else {}
        count_map = counts if isinstance(counts, dict) else {}

        return {
            "AP": self._format_number(metric_map.get("AP", metric_map.get("mAP"))),
            "AP50": self._format_number(metric_map.get("AP50", metric_map.get("mAP50"))),
            "AP75": self._format_number(metric_map.get("AP75")),
            "AR100": self._format_number(metric_map.get("AR100")),
            "precision": self._format_number(
                count_map.get("precision", metric_map.get("precision"))
            ),
            "recall": self._format_number(count_map.get("recall", metric_map.get("recall"))),
        }

    def _extract_compare_per_class(self, payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        per_class = payload.get("per_class")
        if not isinstance(per_class, dict):
            return {}

        extracted: dict[str, dict[str, Any]] = {}
        for class_name, values in per_class.items():
            if not isinstance(values, dict):
                continue
            normalized = str(class_name).strip().lower()
            if not normalized:
                continue
            extracted[normalized] = {
                "__matched_name__": str(class_name).strip(),
                **{key: values.get(key) for key in COMPARE_PER_CLASS_METRIC_KEYS if key in values},
            }
        return extracted

    def _resolve_run_config_reference(self, run_dir: Path) -> tuple[str | None, str | None]:
        snapshot_candidates = (
            run_dir / "config.yaml",
            run_dir / "config_snapshot.yaml",
        )
        snapshot_path = next((path for path in snapshot_candidates if path.is_file()), None)
        if snapshot_path is None:
            return None, "No run-local config snapshot found."

        try:
            snapshot_text = snapshot_path.read_text(encoding="utf-8")
        except OSError:
            return self._relative_path(snapshot_path), "Config snapshot could not be read."

        matches = self._config_text_index().get(snapshot_text, [])
        if len(matches) == 1:
            return self._relative_path(matches[0]), "Matched repo config from the run snapshot."
        if len(matches) > 1:
            return (
                self._relative_path(snapshot_path),
                f"Snapshot matches {len(matches)} repo configs; showing the run-local copy.",
            )
        return self._relative_path(snapshot_path), "Showing the run-local config snapshot."

    def _find_primary_golden_report(self, run_dir: Path) -> str | None:
        golden_path = next(iter(sorted((run_dir / "reports").glob("golden*.json"))), None)
        return self._relative_path(golden_path) if golden_path is not None else None

    def _config_text_index(self) -> dict[str, list[Path]]:
        cached = getattr(self, "_cached_config_text_index", None)
        if cached is not None:
            return cached

        index: dict[str, list[Path]] = {}
        config_root = self.repo_root / "configs"
        if config_root.is_dir():
            for path in sorted(config_root.rglob("*.yaml")):
                try:
                    text = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                index.setdefault(text, []).append(path)
        self._cached_config_text_index = index
        return index

    def _infer_curated_compare_label(self, *values: str | None) -> str | None:
        haystack = " ".join(value for value in values if value).lower()
        best_label: str | None = None
        best_length = -1
        for label, tokens in CURATED_COMPARE_LABELS:
            matching_lengths = [len(token) for token in tokens if token in haystack]
            if not matching_lengths:
                continue
            label_best = max(matching_lengths)
            if label_best > best_length:
                best_label = label
                best_length = label_best
        return best_label

    def _resolve_repo_recorded_path(self, value: str | None) -> Path | None:
        if not value:
            return None
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (self.repo_root / candidate).resolve()
        try:
            candidate.relative_to(self.repo_root)
        except ValueError:
            return None
        return candidate if candidate.exists() else None

    def _infer_split_name_from_path(self, path: Path) -> str:
        stem = path.stem.lower()
        for split_name in ("train", "val", "test"):
            if stem.endswith(f"_{split_name}"):
                return split_name
        return stem

    def _load_dataset_config_references(self) -> list[ConfigReferenceView]:
        config_dir = self.repo_root / "configs"
        patterns = ("balance_*.yaml", "coco_replay_*.yaml", "crop_*.yaml")
        paths = self._glob_many(config_dir, patterns)
        items: list[ConfigReferenceView] = []
        for path in paths:
            payload = self._load_yaml(path)
            targets = [
                self._path_from_config(path, "source_coco", payload.get("source_coco")),
                self._path_from_config(path, "source_images_dir", payload.get("source_images_dir")),
                self._path_from_config(path, "out_dir", payload.get("out_dir")),
            ]
            selection = payload.get("selection", {})
            if isinstance(selection, dict):
                cap = selection.get("max_positive_images_per_class")
                min_side = selection.get("min_bbox_min_side")
                description = (
                    f"selection: min_bbox_min_side={min_side}, max_positive_images_per_class={cap}"
                )
            else:
                description = "Dataset prep config."
            items.append(
                ConfigReferenceView(
                    label=path.name,
                    relative_path=self._relative_path(path),
                    description=description,
                    targets=[target for target in targets if target is not None],
                )
            )
        return items

    def _load_merge_config_references(self) -> list[ConfigReferenceView]:
        config_dir = self.repo_root / "configs"
        items: list[ConfigReferenceView] = []
        for path in sorted(config_dir.glob("merge_*.yaml")):
            payload = self._load_yaml(path)
            raw_sources = payload.get("sources", [])
            targets: list[RepoPathView] = []
            if isinstance(raw_sources, list):
                for index, entry in enumerate(raw_sources[:4], start=1):
                    if not isinstance(entry, dict):
                        continue
                    source_name = str(entry.get("name") or f"source_{index}")
                    coco_path = self._path_from_config(
                        path, f"{source_name}.coco", entry.get("coco")
                    )
                    images_dir = self._path_from_config(
                        path,
                        f"{source_name}.images_dir",
                        entry.get("images_dir"),
                    )
                    contract = self._path_from_config(
                        path,
                        f"{source_name}.contract",
                        entry.get("contract"),
                    )
                    for target in (coco_path, images_dir, contract):
                        if target is not None:
                            targets.append(target)
            description = f"Merge manifest with {len(raw_sources) if isinstance(raw_sources, list) else 0} sources."
            items.append(
                ConfigReferenceView(
                    label=path.name,
                    relative_path=self._relative_path(path),
                    description=description,
                    targets=targets,
                )
            )
        return items

    def _load_training_config_references(self) -> list[ConfigReferenceView]:
        config_dir = self.repo_root / "configs"
        items: list[ConfigReferenceView] = []
        for path in sorted(config_dir.glob("efficientdet_*.yaml")):
            payload = self._load_yaml(path)
            data = payload.get("data", {})
            outputs = payload.get("outputs", {})
            targets: list[RepoPathView] = []
            if isinstance(data, dict):
                for label, value in (
                    ("data.csv", data.get("csv")),
                    ("data.images_dir", data.get("images_dir")),
                    ("data.label_map_json", data.get("label_map_json")),
                ):
                    target = self._path_from_config(path, label, value)
                    if target is not None:
                        targets.append(target)
            if isinstance(outputs, dict):
                for label, value in (
                    ("outputs.work_dir", outputs.get("work_dir")),
                    ("outputs.out_dir", outputs.get("out_dir")),
                ):
                    target = self._path_from_config(path, label, value)
                    if target is not None:
                        targets.append(target)
            variant = ""
            model = payload.get("model", {})
            if isinstance(model, dict):
                variant = str(model.get("variant", "")).strip()
            description = f"EfficientDet config ({variant or 'unspecified variant'})."
            items.append(
                ConfigReferenceView(
                    label=path.name,
                    relative_path=self._relative_path(path),
                    description=description,
                    targets=targets,
                )
            )
        return items

    def _scan_artifact_directories(
        self,
        *,
        root: Path,
        limit: int,
        highlight_patterns: tuple[str, ...],
    ) -> list[ArtifactDirectoryView]:
        if not root.is_dir():
            return []

        directories = [entry for entry in root.iterdir() if entry.is_dir()]
        directories.sort(key=lambda path: path.stat().st_mtime, reverse=True)

        results: list[ArtifactDirectoryView] = []
        for path in directories[:limit]:
            highlights = self._collect_highlights(path, highlight_patterns)
            child_entries = list(path.iterdir())
            results.append(
                ArtifactDirectoryView(
                    label=path.name,
                    relative_path=self._relative_path(path),
                    exists=True,
                    summary=f"{len(child_entries)} top-level entries",
                    modified_at=self._format_mtime(path),
                    highlights=highlights,
                )
            )
        return results

    def _find_primary_dataset_coco(self, dataset_dir: Path) -> Path | None:
        candidates = [
            dataset_dir / "instances_materialized.json",
            dataset_dir / "instances.json",
        ]
        for path in candidates:
            if path.is_file():
                return path
        matches = sorted(dataset_dir.glob("instances*.json"))
        return matches[0] if matches else None

    def _find_split_file(self, dataset_dir: Path) -> Path | None:
        direct = dataset_dir / "splits.json"
        if direct.is_file():
            return direct
        sibling = self.repo_root / "work" / "splits" / dataset_dir.name / "splits.json"
        if sibling.is_file():
            return sibling
        return None

    def _load_split_counts(self, dataset_dir: Path) -> list[CountRowView]:
        split_path = self._find_split_file(dataset_dir)
        if split_path is None:
            return []
        payload = self._load_json(split_path)
        if not isinstance(payload, dict):
            return []
        rows: list[CountRowView] = []
        for split_name in ("train", "val", "test"):
            value = payload.get(split_name)
            if isinstance(value, list):
                rows.append(CountRowView(label=split_name, value=len(value)))
        for key, value in payload.items():
            if key in {"train", "val", "test"}:
                continue
            if isinstance(value, list):
                rows.append(CountRowView(label=str(key), value=len(value)))
        return rows

    def _load_qc_summary(self, dataset_dir: Path) -> list[LabelValueView]:
        qc_path = dataset_dir / "qc_report.json"
        payload = self._load_json(qc_path)
        if not isinstance(payload, dict):
            return []

        rows = self._label_values_from_mapping(payload.get("summary"))
        if rows:
            return rows

        input_payload = payload.get("input")
        if isinstance(input_payload, dict):
            rows.extend(
                self._label_values_from_mapping(
                    input_payload,
                    keys=("images", "annotations", "categories"),
                    prefix="input",
                )
            )
        output_payload = payload.get("output")
        if isinstance(output_payload, dict):
            rows.extend(
                self._label_values_from_mapping(
                    output_payload,
                    keys=("images", "annotations", "categories"),
                    prefix="output",
                )
            )
        if "filtered_small_bbox_annotations" in payload:
            rows.append(
                LabelValueView(
                    label="filtered_small_bbox_annotations",
                    value=str(payload["filtered_small_bbox_annotations"]),
                )
            )
        selected_image_counts = payload.get("selected_image_counts")
        if isinstance(selected_image_counts, dict):
            total = sum(
                int(value) for value in selected_image_counts.values() if isinstance(value, int)
            )
            rows.append(LabelValueView(label="selected_image_total", value=str(total)))
        return rows

    def resolve_fiftyone_target(
        self, *, source_kind: str, relative_path: str
    ) -> FiftyOneLaunchTargetView | None:
        if source_kind == "dataset":
            detail = self.load_dataset_detail(relative_path)
            return detail.fiftyone_target if detail is not None else None
        if source_kind == "eval":
            detail = self.load_eval_detail(relative_path)
            return detail.fiftyone_target if detail is not None else None
        return None

    def _find_config_references_for_prefix(self, relative_prefix: str) -> list[ConfigReferenceView]:
        matches: list[ConfigReferenceView] = []
        prefix = relative_prefix.rstrip("/") + "/"
        for group in self.load_config_groups():
            for item in group.items:
                for target in item.targets:
                    if target.relative_path == relative_prefix or target.relative_path.startswith(
                        prefix
                    ):
                        matches.append(item)
                        break
        return matches

    def _resolve_allowed_path(
        self,
        relative_path: str,
        *,
        required_kind: str,
        allowed_roots: tuple[str, ...],
    ) -> Path | None:
        text = relative_path.strip()
        if not text:
            return None
        candidate = (self.repo_root / text).resolve()
        try:
            candidate.relative_to(self.repo_root)
        except ValueError:
            return None
        if not any(self._relative_path(candidate).startswith(prefix) for prefix in allowed_roots):
            return None
        if required_kind == "dir" and not candidate.is_dir():
            return None
        if required_kind == "file" and not candidate.is_file():
            return None
        return candidate

    def _list_known_files(self, root: Path, limit: int = 16) -> list[RepoPathView]:
        if not root.is_dir():
            return []
        files = [path for path in root.rglob("*") if path.is_file()]
        files.sort()
        return [
            self._path_view_from_absolute(path, label=str(path.relative_to(root)))
            for path in files[:limit]
        ]

    def _linked_report_view(self, path: Path, note: str | None = None) -> LinkedArtifactView:
        return LinkedArtifactView(
            label=path.name,
            relative_path=self._relative_path(path),
            exists=path.exists(),
            note=note,
        )

    def _eval_headline(self, path: Path) -> str | None:
        payload = self._load_json(path)
        metrics = payload.get("metrics") if isinstance(payload, dict) else None
        if not isinstance(metrics, dict):
            return None
        for key in ("mAP50", "mAP", "AP50", "AP", "precision", "recall"):
            if key in metrics:
                return f"{key}={self._format_number(metrics[key])}"
        for key, value in metrics.items():
            return f"{key}={self._format_number(value)}"
        return None

    def _golden_headline(self, path: Path) -> str | None:
        payload = self._load_json(path)
        detections = payload.get("detections") if isinstance(payload, dict) else None
        if isinstance(detections, list):
            return f"detections={len(detections)}"
        return None

    def _load_sibling_eval_reports(self, report_path: Path) -> list[LinkedArtifactView]:
        parent = report_path.parent
        siblings: list[LinkedArtifactView] = []
        for path in sorted(parent.glob("eval*.json")):
            if path == report_path:
                continue
            siblings.append(self._linked_report_view(path, note=self._eval_headline(path)))
        return siblings[:8]

    def _infer_run_path(self, path: Path) -> str | None:
        relative = self._relative_path(path)
        parts = Path(relative).parts
        if len(parts) >= 3 and parts[0] == "work" and parts[1] == "runs":
            return str(Path(*parts[:3]))
        return None

    def _label_values_from_mapping(
        self,
        value: Any,
        *,
        keys: tuple[str, ...] | None = None,
        prefix: str | None = None,
    ) -> list[LabelValueView]:
        if not isinstance(value, dict):
            return []
        rows: list[LabelValueView] = []
        if keys is None:
            iterable = value.items()
        else:
            iterable = ((key, value.get(key)) for key in keys if key in value)
        for key, item in iterable:
            text = self._display_value(item)
            if text is None:
                continue
            label = f"{prefix}.{key}" if prefix else str(key)
            rows.append(LabelValueView(label=label, value=text))
        return rows

    def _metric_rows(self, value: Any) -> list[MetricRowView]:
        if not isinstance(value, dict):
            return []
        return [
            MetricRowView(key=str(key), value=self._format_number(item))
            for key, item in value.items()
        ]

    def _per_class_rows(self, value: Any) -> tuple[list[str], list[PerClassMetricRowView]]:
        if not isinstance(value, dict):
            return [], []

        headers: list[str] = []
        for item in value.values():
            if not isinstance(item, dict):
                continue
            for key in item:
                if key not in headers:
                    headers.append(str(key))

        rows: list[PerClassMetricRowView] = []
        for class_name, item in value.items():
            if not isinstance(item, dict):
                continue
            rows.append(
                PerClassMetricRowView(
                    class_name=str(class_name),
                    values={header: self._format_number(item.get(header)) for header in headers},
                )
            )
        return headers, rows

    def _read_excerpt(self, path: Path, line_limit: int = 40) -> str:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return ""
        lines = text.splitlines()
        excerpt = "\n".join(lines[:line_limit])
        if len(lines) > line_limit:
            excerpt += "\n..."
        return excerpt

    def _load_json(self, path: Path | None) -> dict[str, Any]:
        if path is None or not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    @staticmethod
    def _display_value(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return f"{len(value)} keys"
        if isinstance(value, list):
            return f"{len(value)} items"
        return str(value)

    @staticmethod
    def _format_number(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        if value is None:
            return "-"
        return str(value)

    @staticmethod
    def _format_bbox(value: Any) -> str:
        if not isinstance(value, list):
            return "-"
        return "[" + ", ".join(RepositoryReader._format_number(item) for item in value) + "]"

    def _collect_highlights(self, root: Path, patterns: tuple[str, ...]) -> list[str]:
        seen: set[Path] = set()
        highlights: list[str] = []
        for pattern in patterns:
            for match in sorted(root.glob(pattern)):
                if not match.exists() or match in seen:
                    continue
                seen.add(match)
                highlights.append(str(match.relative_to(root)))
                if len(highlights) >= 6:
                    return highlights
        return highlights

    def _build_repo_path(
        self, label: str, relative_path: str, note: str | None = None
    ) -> RepoPathView:
        return self._path_view_from_absolute(self.repo_root / relative_path, label=label, note=note)

    def _build_dataset_fiftyone_target(
        self,
        *,
        dataset_dir: Path,
        primary_coco: Path | None,
        images_dir: Path,
    ) -> FiftyOneLaunchTargetView:
        dataset_name = f"owli-{dataset_dir.name}"
        if primary_coco is None:
            return FiftyOneLaunchTargetView(
                source_kind="dataset",
                source_path=self._relative_path(dataset_dir),
                source_label="Dataset detail",
                back_path=self._relative_path(dataset_dir),
                back_label="Back to dataset detail",
                back_route_name="dataset_detail_page",
                title=f"Open {dataset_dir.name} in FiftyOne",
                dataset_name=dataset_name,
                coco_path=None,
                images_dir=None,
                can_launch=False,
                message="No COCO file was found for this dataset directory.",
            )
        if not images_dir.is_dir():
            return FiftyOneLaunchTargetView(
                source_kind="dataset",
                source_path=self._relative_path(dataset_dir),
                source_label="Dataset detail",
                back_path=self._relative_path(dataset_dir),
                back_label="Back to dataset detail",
                back_route_name="dataset_detail_page",
                title=f"Open {dataset_dir.name} in FiftyOne",
                dataset_name=dataset_name,
                coco_path=self._relative_path(primary_coco),
                images_dir=None,
                can_launch=False,
                message=(
                    "This dataset is not ready for FiftyOne yet because no local images "
                    "directory was found at <dataset>/images."
                ),
            )
        return FiftyOneLaunchTargetView(
            source_kind="dataset",
            source_path=self._relative_path(dataset_dir),
            source_label="Dataset detail",
            back_path=self._relative_path(dataset_dir),
            back_label="Back to dataset detail",
            back_route_name="dataset_detail_page",
            title=f"Open {dataset_dir.name} in FiftyOne",
            dataset_name=dataset_name,
            coco_path=self._relative_path(primary_coco),
            images_dir=self._relative_path(images_dir),
            can_launch=True,
            message=(
                "Supports local COCO datasets with a repo-visible images directory under "
                "<dataset>/images."
            ),
        )

    def _build_eval_fiftyone_target(
        self,
        *,
        report_path: Path,
        payload: dict[str, Any],
    ) -> FiftyOneLaunchTargetView:
        run_path = self._infer_run_path(report_path)
        back_path = run_path or self._relative_path(report_path)
        back_label = "Back to run detail" if run_path else "Back to eval detail"
        back_route_name = "run_detail_page" if run_path else "eval_detail_page"
        default_name = f"owli-{report_path.stem}"
        coco_path = self._resolve_runtime_repo_path(payload.get("coco_path"))
        images_dir = self._resolve_runtime_repo_path(payload.get("images_dir"))
        if coco_path is None:
            return FiftyOneLaunchTargetView(
                source_kind="eval",
                source_path=self._relative_path(report_path),
                source_label="Eval detail",
                back_path=back_path,
                back_label=back_label,
                back_route_name=back_route_name,
                title=f"Open eval dataset for {report_path.name} in FiftyOne",
                dataset_name=default_name,
                coco_path=None,
                images_dir=None,
                can_launch=False,
                message="This eval report does not expose a usable COCO path.",
            )
        if images_dir is None or not images_dir.is_dir():
            return FiftyOneLaunchTargetView(
                source_kind="eval",
                source_path=self._relative_path(report_path),
                source_label="Eval detail",
                back_path=back_path,
                back_label=back_label,
                back_route_name=back_route_name,
                title=f"Open eval dataset for {report_path.name} in FiftyOne",
                dataset_name=default_name,
                coco_path=self._relative_path(coco_path),
                images_dir=self._relative_path(images_dir) if images_dir is not None else None,
                can_launch=False,
                message="This eval report does not point to a usable local images directory.",
            )
        return FiftyOneLaunchTargetView(
            source_kind="eval",
            source_path=self._relative_path(report_path),
            source_label="Eval detail",
            back_path=back_path,
            back_label=back_label,
            back_route_name=back_route_name,
            title=f"Open eval dataset for {report_path.name} in FiftyOne",
            dataset_name=default_name,
            coco_path=self._relative_path(coco_path),
            images_dir=self._relative_path(images_dir),
            can_launch=True,
            message="Uses the COCO and images paths recorded in the eval JSON report.",
        )

    def _path_from_config(self, config_path: Path, label: str, value: Any) -> RepoPathView | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        raw = Path(text)
        resolved = raw if raw.is_absolute() else (config_path.parent / raw).resolve()
        return self._path_view_from_absolute(resolved, label=label)

    def _path_view_from_absolute(
        self, path: Path, *, label: str, note: str | None = None
    ) -> RepoPathView:
        exists = path.exists()
        kind = "dir" if path.is_dir() else "file"
        detail: str | None = None
        if exists:
            if path.is_dir():
                try:
                    detail = f"{sum(1 for _ in path.iterdir())} entries"
                except OSError:
                    detail = None
            else:
                try:
                    detail = f"{path.stat().st_size} bytes"
                except OSError:
                    detail = None
        return RepoPathView(
            label=label,
            relative_path=self._relative_path(path),
            exists=exists,
            kind=kind,
            note=note,
            modified_at=self._format_mtime(path),
            detail=detail,
        )

    def _relative_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.repo_root))
        except ValueError:
            return str(path.resolve())

    def _resolve_runtime_repo_path(self, value: Any) -> Path | None:
        text = self._optional_text(value)
        if text is None:
            return None
        raw = Path(text)
        candidate = raw if raw.is_absolute() else (self.repo_root / raw)
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.repo_root)
        except ValueError:
            return None
        return resolved if resolved.exists() else None

    def _format_mtime(self, path: Path) -> str | None:
        if not path.exists():
            return None
        try:
            return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        except OSError:
            return None

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            return {}
        return payload

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _as_mapping(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        return value

    @staticmethod
    def _as_list_of_strings(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _glob_many(root: Path, patterns: tuple[str, ...]) -> list[Path]:
        matched: set[Path] = set()
        for pattern in patterns:
            matched.update(root.glob(pattern))
        return sorted(matched)
