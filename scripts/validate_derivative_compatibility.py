"""Validate a derivative layer's declared compatibility with this RAG Platform base."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any


_FIELDS = frozenset({"schema_version", "base_project", "base_revision", "base_api_major"})
_REVISION = re.compile(r"(?:unrecorded|[0-9a-f]{7,64})")
# ``rag-studio`` is the published identity before the repository rebrand.  It
# remains valid for schema v1 manifests so existing derived projects can
# upgrade deliberately rather than being rejected by a cosmetic rename.
_SUPPORTED_BASE_PROJECTS = frozenset({"rag-platform", "rag-studio"})


class DerivativeCompatibilityError(ValueError):
    """The derivative compatibility declaration is unsafe or stale."""


def validate_compatibility(path: Path, *, base_root: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise DerivativeCompatibilityError("compatibility manifest cannot be read") from error
    if not isinstance(payload, dict) or frozenset(payload) != _FIELDS:
        raise DerivativeCompatibilityError("compatibility manifest fields are invalid")
    if payload["schema_version"] != 1 or payload["base_project"] not in _SUPPORTED_BASE_PROJECTS:
        raise DerivativeCompatibilityError("compatibility manifest base is unsupported")
    if not isinstance(payload["base_revision"], str) or _REVISION.fullmatch(payload["base_revision"]) is None:
        raise DerivativeCompatibilityError("compatibility manifest revision is invalid")
    if payload["base_api_major"] != 2:
        raise DerivativeCompatibilityError("derivative requires a different API major")
    try:
        project = tomllib.loads((base_root / "pyproject.toml").read_text(encoding="utf-8"))
        version = project["project"]["version"]
    except (KeyError, OSError, tomllib.TOMLDecodeError) as error:
        raise DerivativeCompatibilityError("base package metadata is unavailable") from error
    if not isinstance(version, str) or not version.startswith("2."):
        raise DerivativeCompatibilityError("base package major is incompatible")
    uses_legacy_identity = payload["base_project"] == "rag-studio"
    return {
        "base_version": version,
        "base_revision": payload["base_revision"],
        "compatible": True,
        "identity_upgrade_available": uses_legacy_identity,
    }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--base-root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args(argv)
    try:
        result = validate_compatibility(arguments.manifest, base_root=arguments.base_root.resolve())
    except DerivativeCompatibilityError as error:
        print(f"derivative compatibility rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
