from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RepoPathView:
    label: str
    relative_path: str
    exists: bool
    kind: str
    note: str | None = None
    modified_at: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ArtifactDirectoryView:
    label: str
    relative_path: str
    exists: bool
    summary: str
    modified_at: str | None = None
    highlights: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContractRoleView:
    name: str
    class_names: list[str]


@dataclass(frozen=True)
class ContractClassView:
    order: int
    name: str
    role: str
    rationale: str | None = None


@dataclass(frozen=True)
class ContractView:
    key: str
    title: str
    relative_path: str
    exists: bool
    status: str | None
    purpose: str | None
    class_count: int
    roles: list[ContractRoleView]
    classes: list[ContractClassView]
    out_of_scope: list[str]
    modified_at: str | None = None


@dataclass(frozen=True)
class ConfigReferenceView:
    label: str
    relative_path: str
    description: str
    targets: list[RepoPathView]


@dataclass(frozen=True)
class ConfigGroupView:
    title: str
    description: str
    items: list[ConfigReferenceView]


@dataclass(frozen=True)
class RepositoryViewModel:
    repo_root: str
    docs: list[RepoPathView]
    contracts: list[ContractView]
    artifact_roots: list[RepoPathView]
    recent_datasets: list[ArtifactDirectoryView]
    recent_runs: list[ArtifactDirectoryView]
    config_groups: list[ConfigGroupView]


@dataclass(frozen=True)
class JobArtifactView:
    label: str
    relative_path: str
    exists: bool


@dataclass(frozen=True)
class JobView:
    job_id: str
    job_type: str
    title: str
    status: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    exit_code: int | None
    command_preview: str
    log_path: str
    parameters: dict[str, object] = field(default_factory=dict)
    artifacts: list[JobArtifactView] = field(default_factory=list)
    log_text: str | None = None


@dataclass(frozen=True)
class LabelValueView:
    label: str
    value: str


@dataclass(frozen=True)
class CountRowView:
    label: str
    value: int


@dataclass(frozen=True)
class MetricRowView:
    key: str
    value: str


@dataclass(frozen=True)
class PerClassMetricRowView:
    class_name: str
    values: dict[str, str]


@dataclass(frozen=True)
class LinkedArtifactView:
    label: str
    relative_path: str
    exists: bool
    note: str | None = None


@dataclass(frozen=True)
class FiftyOneLaunchTargetView:
    source_kind: str
    source_path: str
    source_label: str
    back_path: str
    back_label: str
    back_route_name: str
    title: str
    dataset_name: str | None
    coco_path: str | None
    images_dir: str | None
    can_launch: bool
    message: str


@dataclass(frozen=True)
class FiftyOneLaunchResultView:
    status: str
    message: str
    app_url: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class DatasetDetailView:
    title: str
    relative_path: str
    primary_coco_path: str | None
    images_dir: str | None
    fiftyone_target: FiftyOneLaunchTargetView | None
    summary: list[LabelValueView]
    class_distribution: list[CountRowView]
    split_counts: list[CountRowView]
    qc_summary: list[LabelValueView]
    related_files: list[RepoPathView]
    related_configs: list[ConfigReferenceView]


@dataclass(frozen=True)
class RunDetailView:
    run_id: str
    relative_path: str
    summary: list[LabelValueView]
    config_files: list[RepoPathView]
    artifact_files: list[RepoPathView]
    report_files: list[RepoPathView]
    eval_reports: list[LinkedArtifactView]
    golden_reports: list[LinkedArtifactView]


@dataclass(frozen=True)
class CompareRunOptionView:
    run_id: str
    relative_path: str
    selected: bool
    eval_count: int
    curated_label: str | None = None


@dataclass(frozen=True)
class CompareTargetOptionView:
    key: str
    label: str
    count: int
    selected: bool


@dataclass(frozen=True)
class CompareRowView:
    run_id: str
    run_relative_path: str
    run_display_name: str
    curated_label: str | None
    stage_label: str | None
    dataset_label: str
    config_path: str | None
    config_note: str | None
    eval_relative_path: str
    eval_label: str
    golden_relative_path: str | None
    metrics: dict[str, str]


@dataclass(frozen=True)
class RunsCompareView:
    selected_target_key: str | None
    selected_target_label: str | None
    target_options: list[CompareTargetOptionView]
    run_options: list[CompareRunOptionView]
    metric_keys: list[str]
    rows: list[CompareRowView]
    selection_summary: str
    scope_note: str


@dataclass(frozen=True)
class EvalDetailView:
    title: str
    relative_path: str
    summary: list[LabelValueView]
    fiftyone_target: FiftyOneLaunchTargetView | None
    metrics: list[MetricRowView]
    summary_counts: list[MetricRowView]
    per_class_headers: list[str]
    per_class_rows: list[PerClassMetricRowView]
    related_run_path: str | None
    sibling_reports: list[LinkedArtifactView]
    raw_excerpt: str | None = None


@dataclass(frozen=True)
class GoldenDetectionView:
    class_name: str
    score: str
    bbox: str


@dataclass(frozen=True)
class GoldenDetailView:
    title: str
    relative_path: str
    summary: list[LabelValueView]
    contract: list[LabelValueView]
    model_metadata: list[LabelValueView]
    inspect_tflite: list[LabelValueView]
    detections: list[GoldenDetectionView]
    related_run_path: str | None
