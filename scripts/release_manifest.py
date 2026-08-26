"""Create a non-sensitive release-input manifest for build and rollback review."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.provenance import (  # noqa: E402
    inspect_source_provenance,
    require_clean_source,
)

_INPUTS = (
    "Dockerfile",
    "compose.yaml",
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock.in",
    "requirements-py311.lock",
    "requirements-py312.lock",
    "requirements-dev.txt",
)


def release_manifest(*, root: Path = PROJECT_ROOT, generated_at: datetime | None = None) -> dict[str, Any]:
    """Describe immutable source inputs without reading environment or secret files."""

    resolved_root = root.resolve()
    timestamp = generated_at or datetime.now(UTC)
    files = {relative: _sha256(resolved_root / relative) for relative in _INPUTS}
    provenance = inspect_source_provenance(resolved_root)
    return {
        "schema_version": 1,
        "generated_at": timestamp.isoformat(),
        "source_revision": provenance.revision,
        "working_tree_clean": provenance.working_tree_clean,
        "package_version": _package_version(resolved_root),
        "build_inputs": files,
    }


def require_clean(manifest: dict[str, Any]) -> None:
    from rag_system.provenance import SourceProvenance

    require_clean_source(
        SourceProvenance(
            revision=manifest["source_revision"],
            working_tree_clean=manifest["working_tree_clean"],
        ),
        artifact="a release manifest",
    )


def _sha256(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"release input cannot be read: {path.name}") from error
    return hashlib.sha256(payload).hexdigest()


def _package_version(root: Path) -> str:
    try:
        import tomllib

        payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        version = payload["project"]["version"]
    except (KeyError, OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("pyproject project version is unavailable") from error
    if not isinstance(version, str) or not version.strip():
        raise ValueError("pyproject project version is invalid")
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--require-clean", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        manifest = release_manifest()
        if arguments.require_clean:
            require_clean(manifest)
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if arguments.json_output:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
