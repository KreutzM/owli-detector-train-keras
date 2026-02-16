from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from owli_train.data.coco import load_coco, validate_coco
from owli_train.data.merge_coco import MergeManifest, load_merge_manifest


class MaterializeImagesError(RuntimeError):
    """Raised when image materialization fails."""


@dataclass(frozen=True)
class MaterializeImagesArtifacts:
    out_images_dir: Path
    out_coco_path: Path
    images_total: int
    images_written: int
    images_skipped: int
    copied: int
    symlinked: int


def _normalize_rel_file_name(raw: Any) -> str:
    text = str(raw).strip().replace("\\", "/")
    if not text:
        raise MaterializeImagesError("Encountered empty image file_name in COCO images list.")
    rel = PurePosixPath(text)
    if rel.is_absolute():
        raise MaterializeImagesError(f"COCO image file_name must be relative, got: {text}")
    if ".." in rel.parts:
        raise MaterializeImagesError(f"COCO image file_name must not contain '..': {text}")
    return rel.as_posix()


def _candidate_from_manifest_source(file_name: str, source: Any) -> Path | None:
    images_dir = getattr(source, "images_dir", None)
    if images_dir is None:
        return None

    rel = file_name
    raw_prefix = str(getattr(source, "file_name_prefix", "") or "").strip().strip("/")
    if raw_prefix:
        prefix = raw_prefix.replace("\\", "/")
        marker = prefix + "/"
        if rel == prefix or not rel.startswith(marker):
            return None
        rel = rel[len(marker) :]
    return Path(images_dir) / rel


def _resolve_source_path(
    *,
    file_name: str,
    manifest: MergeManifest | None,
    source_images_dirs: list[Path],
) -> Path:
    candidates: list[Path] = []

    if manifest is not None:
        for source in manifest.sources:
            candidate = _candidate_from_manifest_source(file_name, source)
            if candidate is None:
                continue
            if candidate.is_file():
                candidates.append(candidate)

    for root in source_images_dirs:
        candidate = root / file_name
        if candidate.is_file():
            candidates.append(candidate)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate)

    if not unique:
        raise MaterializeImagesError(f"Could not resolve source image for merged file_name: {file_name}")
    if len(unique) > 1:
        preview = ", ".join(str(path) for path in unique[:5])
        suffix = "" if len(unique) <= 5 else ", ..."
        raise MaterializeImagesError(
            "Ambiguous source image for merged file_name "
            f"{file_name}: {preview}{suffix}. Use sources[].file_name_prefix in merge manifest."
        )
    return unique[0]


def _write_link_or_copy(*, src: Path, dst: Path, mode: str) -> tuple[int, int]:
    if mode not in {"auto", "symlink", "copy"}:
        raise MaterializeImagesError("mode must be one of: auto, symlink, copy")

    if mode in {"auto", "symlink"}:
        try:
            os.symlink(src.resolve(), dst)
            return 0, 1
        except OSError as exc:
            if mode == "symlink":
                raise MaterializeImagesError(
                    f"Failed to create symlink for {dst} -> {src}: {exc}"
                ) from exc

    shutil.copy2(src, dst)
    return 1, 0


def materialize_coco_images(
    *,
    coco_path: Path,
    out_images_dir: Path,
    out_coco_path: Path,
    merge_manifest_path: Path | None = None,
    source_images_dirs: list[Path] | None = None,
    mode: str = "auto",
    overwrite: bool = False,
) -> MaterializeImagesArtifacts:
    coco = load_coco(coco_path)
    validate_coco(coco)

    if source_images_dirs is None:
        source_images_dirs = []

    for path in source_images_dirs:
        if not path.is_dir():
            raise MaterializeImagesError(f"source_images_dir was not found: {path}")

    manifest: MergeManifest | None = None
    if merge_manifest_path is not None:
        manifest = load_merge_manifest(merge_manifest_path)

    if manifest is None and not source_images_dirs:
        raise MaterializeImagesError(
            "Provide either --merge-manifest or at least one --source-images-dir."
        )

    out_images_dir.mkdir(parents=True, exist_ok=True)

    images_written = 0
    images_skipped = 0
    copied = 0
    symlinked = 0

    normalized_images: list[dict[str, Any]] = []
    for image in coco.get("images", []):
        item = dict(image)
        file_name = _normalize_rel_file_name(item.get("file_name"))
        src = _resolve_source_path(
            file_name=file_name,
            manifest=manifest,
            source_images_dirs=source_images_dirs,
        )

        dst = out_images_dir / file_name
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            if overwrite:
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            else:
                images_skipped += 1
                item["file_name"] = file_name
                normalized_images.append(item)
                continue

        copied_delta, symlink_delta = _write_link_or_copy(src=src, dst=dst, mode=mode)
        copied += copied_delta
        symlinked += symlink_delta
        images_written += 1

        item["file_name"] = file_name
        normalized_images.append(item)

    out_payload = dict(coco)
    out_payload["images"] = normalized_images

    out_coco_path.parent.mkdir(parents=True, exist_ok=True)
    out_coco_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    validate_coco(out_payload, images_dir=out_images_dir)

    return MaterializeImagesArtifacts(
        out_images_dir=out_images_dir,
        out_coco_path=out_coco_path,
        images_total=len(normalized_images),
        images_written=images_written,
        images_skipped=images_skipped,
        copied=copied,
        symlinked=symlinked,
    )
