"""Generate a deterministic SPDX 2.3 runtime dependency inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement


_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9_.+!-]*)$")
_NORMALIZE = re.compile(r"[-_.]+")


class SbomError(ValueError):
    """The declared runtime environment cannot produce a trustworthy SBOM."""


@dataclass(frozen=True, slots=True)
class Distribution:
    name: str
    version: str
    requires: tuple[str, ...]


def load_pins(requirements_path: Path) -> dict[str, str]:
    """Read only exact, direct pins; recursive requirement files are rejected."""

    try:
        lines = requirements_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise SbomError("runtime requirements cannot be read") from error
    pins: dict[str, str] = {}
    for raw in lines:
        value = raw.split("#", 1)[0].strip()
        if not value:
            continue
        match = _PIN.fullmatch(value)
        if match is None:
            raise SbomError("runtime requirements must contain only exact direct pins")
        name = _normalized_name(match.group(1))
        if name in pins:
            raise SbomError("runtime requirements contain duplicate package pins")
        pins[name] = match.group(2)
    if not pins:
        raise SbomError("runtime requirements are empty")
    return pins


def installed_distributions() -> dict[str, Distribution]:
    result: dict[str, Distribution] = {}
    for item in metadata.distributions():
        name = item.metadata.get("Name")
        if not isinstance(name, str) or not name.strip():
            continue
        normalized = _normalized_name(name)
        if normalized in result:
            raise SbomError("installed environment contains duplicate package metadata")
        result[normalized] = Distribution(
            name=name.strip(),
            version=item.version,
            requires=tuple(item.requires or ()),
        )
    return result


def build_spdx(
    pins: Mapping[str, str],
    installed: Mapping[str, Distribution],
    *,
    project_name: str,
) -> dict[str, Any]:
    """Build an SPDX document from direct pins and their installed dependency closure."""

    if not pins:
        raise SbomError("runtime requirements are empty")
    for name, version in pins.items():
        item = installed.get(name)
        if item is None or item.version != version:
            raise SbomError(f"declared runtime pin is not installed: {name}=={version}")

    reachable: set[str] = set()
    edges: set[tuple[str, str]] = set()
    pending: deque[str] = deque(sorted(pins))
    while pending:
        name = pending.popleft()
        if name in reachable:
            continue
        item = installed.get(name)
        if item is None:
            continue
        reachable.add(name)
        for dependency in item.requires:
            dependent_name = _dependency_name(dependency)
            if dependent_name is None or dependent_name not in installed:
                continue
            edges.add((name, dependent_name))
            pending.append(dependent_name)

    package_id = {name: _spdx_id(name) for name in reachable}
    packages = [
        {
            "SPDXID": package_id[name],
            "name": installed[name].name,
            "versionInfo": installed[name].version,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "supplier": "NOASSERTION",
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:pypi/{name}@{installed[name].version}",
                }
            ],
        }
        for name in sorted(reachable)
    ]
    digest = hashlib.sha256(
        "\n".join(f"{name}=={installed[name].version}" for name in sorted(reachable)).encode("utf-8")
    ).hexdigest()
    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": package_id[name],
        }
        for name in sorted(reachable)
    ]
    relationships.extend(
        {
            "spdxElementId": package_id[parent],
            "relationshipType": "DEPENDS_ON",
            "relatedSpdxElement": package_id[child],
        }
        for parent, child in sorted(edges)
        if parent in package_id and child in package_id
    )
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{project_name} runtime dependencies",
        "documentNamespace": f"https://spdx.org/spdxdocs/{_normalized_name(project_name)}-{digest}",
        "creationInfo": {
            "created": "1970-01-01T00:00:00Z",
            "creators": ["Tool: rag-system-generate-runtime-sbom"],
        },
        "documentDescribes": [package_id[name] for name in sorted(reachable)],
        "packages": packages,
        "relationships": relationships,
    }


def _normalized_name(value: str) -> str:
    return _NORMALIZE.sub("-", value).lower()


def _dependency_name(value: str) -> str | None:
    try:
        requirement = Requirement(value)
    except InvalidRequirement as error:
        raise SbomError("installed dependency metadata is invalid") from error
    environment = default_environment()
    environment["extra"] = ""
    if requirement.marker is not None and not requirement.marker.evaluate(environment):
        return None
    return _normalized_name(requirement.name)


def _spdx_id(name: str) -> str:
    return "SPDXRef-Package-" + re.sub(r"[^A-Za-z0-9.-]", "-", name)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=Path("requirements.txt"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-name", default="RAG Platform")
    arguments = parser.parse_args(argv)
    try:
        document = build_spdx(
            load_pins(arguments.requirements),
            installed_distributions(),
            project_name=arguments.project_name,
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, SbomError) as error:
        print(f"runtime SBOM rejected: {error}", file=sys.stderr)
        return 2
    print(f"runtime SBOM written: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
