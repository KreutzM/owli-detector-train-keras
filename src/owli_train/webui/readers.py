from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from owli_train.webui.models import (
    ArtifactDirectoryView,
    ConfigGroupView,
    ConfigReferenceView,
    ContractClassView,
    ContractRoleView,
    ContractView,
    RepoPathView,
    RepositoryViewModel,
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
    ("Splits root", "work/splits", "Deterministic split artifacts."),
    ("Reports root", "work/reports", "Evaluation and inspection reports."),
    ("Outputs root", "outputs", "Model outputs configured by training backends."),
    ("Data root", "data", "Raw or imported source data."),
)

CONTRACT_ORDER: tuple[str, ...] = ("ba_v1", "ba_v2_hazard")


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
                "config_snapshot.yaml",
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
