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
