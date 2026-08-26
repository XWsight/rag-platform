"""Require a stable package version and matching Git tag before creating a release."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VERSION = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)$")


def package_version(*, root: Path = PROJECT_ROOT) -> str:
    try:
        payload = tomllib.loads((root.resolve() / "pyproject.toml").read_text(encoding="utf-8"))
        version = payload["project"]["version"]
    except (KeyError, OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError("project package version is unavailable") from error
    if not isinstance(version, str) or _VERSION.fullmatch(version) is None:
        raise ValueError("project package version must be a stable semantic version")
    return version


def verify_release_tag(tag: str, *, version: str) -> None:
    if tag != f"v{version}":
        raise ValueError("release tag must exactly match the package version, for example v2.0.0")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Git tag expected to match v<package-version>.")
    arguments = parser.parse_args(argv)
    try:
        version = package_version()
        verify_release_tag(arguments.tag, version=version)
    except ValueError as error:
        parser.error(str(error))
    print(f"Release tag verified: {arguments.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
