"""Copy one verified Docker data volume to a different current volume name."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass


_VOLUME_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_BUSYBOX_IMAGE = "busybox:1.37.0"


class VolumeMigrationError(RuntimeError):
    """Raised when a volume copy cannot be proven safe."""


@dataclass(frozen=True, slots=True)
class VolumeMigrationPlan:
    """Validated, reviewable commands for one non-overwriting volume copy."""

    source_volume: str
    destination_volume: str

    @property
    def create_command(self) -> tuple[str, ...]:
        return ("docker", "volume", "create", self.destination_volume)

    @property
    def copy_command(self) -> tuple[str, ...]:
        return (
            "docker",
            "run",
            "--rm",
            "-v",
            f"{self.source_volume}:/source:ro",
            "-v",
            f"{self.destination_volume}:/destination",
            _BUSYBOX_IMAGE,
            "sh",
            "-ec",
            "cd /source && tar -cf - . | tar -C /destination -xf -",
        )

    def manifest_command(self, volume_name: str) -> tuple[str, ...]:
        return (
            "docker",
            "run",
            "--rm",
            "-v",
            f"{volume_name}:/data:ro",
            _BUSYBOX_IMAGE,
            "sh",
            "-ec",
            "cd /data && find . -type f -exec sha256sum {} \\; | LC_ALL=C sort",
        )


def build_plan(*, source_volume: str, destination_volume: str) -> VolumeMigrationPlan:
    """Validate names before displaying or executing any Docker command."""

    for label, value in (("source_volume", source_volume), ("destination_volume", destination_volume)):
        if not isinstance(value, str) or _VOLUME_NAME.fullmatch(value) is None:
            raise VolumeMigrationError(f"{label} is not a safe Docker volume name")
    if source_volume == destination_volume:
        raise VolumeMigrationError("source and destination volumes must differ")
    return VolumeMigrationPlan(source_volume=source_volume, destination_volume=destination_volume)


def execute_plan(plan: VolumeMigrationPlan) -> dict[str, object]:
    """Copy once, then compare file hashes without deleting either volume."""

    _require_volume(plan.source_volume, must_exist=True)
    _require_volume(plan.destination_volume, must_exist=False)
    _run(plan.create_command)
    try:
        _run(plan.copy_command)
        source_manifest = _run(plan.manifest_command(plan.source_volume))
        destination_manifest = _run(plan.manifest_command(plan.destination_volume))
    except VolumeMigrationError as error:
        raise VolumeMigrationError(
            "migration did not complete; the destination volume was preserved for inspection"
        ) from error
    if source_manifest != destination_manifest:
        raise VolumeMigrationError(
            "migration verification failed; the destination volume was preserved for inspection"
        )
    return {
        "source_volume": plan.source_volume,
        "destination_volume": plan.destination_volume,
        "file_count": len(source_manifest.splitlines()),
        "manifest_sha256": hashlib.sha256(source_manifest.encode("utf-8")).hexdigest(),
        "verified": True,
    }


def _require_volume(volume_name: str, *, must_exist: bool) -> None:
    try:
        result = subprocess.run(
            ("docker", "volume", "inspect", volume_name),
            check=False,
            capture_output=True,
            encoding="utf-8",
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise VolumeMigrationError("Docker could not inspect the requested volume") from error
    exists = result.returncode == 0
    if exists != must_exist:
        condition = "does not exist" if must_exist else "already exists"
        raise VolumeMigrationError(f"{volume_name} {condition}")
    if result.returncode not in {0, 1}:
        raise VolumeMigrationError("Docker could not inspect the requested volume")


def _run(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            tuple(command), check=False, capture_output=True, encoding="utf-8", timeout=300
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise VolumeMigrationError("Docker command could not be completed") from error
    if result.returncode != 0:
        raise VolumeMigrationError("Docker command failed")
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-volume", required=True)
    parser.add_argument("--destination-volume", required=True)
    parser.add_argument("--execute", action="store_true", help="Create and copy after reviewing plan.")
    arguments = parser.parse_args(argv)
    try:
        plan = build_plan(
            source_volume=arguments.source_volume,
            destination_volume=arguments.destination_volume,
        )
        if not arguments.execute:
            print(
                json.dumps(
                    {
                        "source_volume": plan.source_volume,
                        "destination_volume": plan.destination_volume,
                        "commands": [list(plan.create_command), list(plan.copy_command)],
                        "execute_required": True,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        print(json.dumps(execute_plan(plan), ensure_ascii=False, sort_keys=True))
        return 0
    except VolumeMigrationError as error:
        print(f"volume migration rejected: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
