"""Capture the non-sensitive Git identity of locally generated artifacts."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


_REVISION_PATTERN = re.compile(r"[0-9a-f]{7,64}")


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """The source identity available to a local build, benchmark, or report."""

    revision: str | None
    working_tree_clean: bool | None


def inspect_source_provenance(root: Path) -> SourceProvenance:
    """Return Git revision and cleanliness without reading source contents."""

    resolved_root = root.resolve()
    revision_result = _git(resolved_root, "rev-parse", "--verify", "HEAD")
    revision = None
    if revision_result is not None:
        candidate = revision_result.stdout.strip()
        if _REVISION_PATTERN.fullmatch(candidate) is not None:
            revision = candidate

    status_result = _git(
        resolved_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    clean = None if status_result is None else not bool(status_result.stdout.strip())
    return SourceProvenance(revision=revision, working_tree_clean=clean)


def require_clean_source(provenance: SourceProvenance, *, artifact: str) -> None:
    """Reject a source tree that cannot identify an immutable artifact input."""

    if provenance.revision is None:
        raise ValueError(f"a Git source revision is required for {artifact}")
    if provenance.working_tree_clean is not True:
        raise ValueError(f"a clean Git working tree is required for {artifact}")


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            encoding="utf-8",
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None


__all__ = ["SourceProvenance", "inspect_source_provenance", "require_clean_source"]
