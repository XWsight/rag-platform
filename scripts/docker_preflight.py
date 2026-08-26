"""Verify Docker Engine and this repository's Compose prerequisites without starting a stack."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RunCommand = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class DockerPreflightError(RuntimeError):
    """Docker is unavailable or the local Compose prerequisites are incomplete."""


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DockerPreflightError("Docker CLI could not be executed") from error


def verify_docker_engine(*, run: RunCommand = _run) -> str:
    result = run(("docker", "version", "--format", "{{.Server.Version}}"))
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    details = (result.stderr or result.stdout).lower()
    if "dockerdesktoplinuxengine" in details:
        raise DockerPreflightError(
            "Docker Desktop Linux engine is unavailable. Start Docker Desktop; if it reports a "
            "stale local socket, close it and repair only the named path shown by Docker before retrying."
        )
    raise DockerPreflightError("Docker Engine is unavailable; inspect Docker Desktop or the host service.")


def verify_compose_prerequisites(*, root: Path = PROJECT_ROOT, run: RunCommand = _run) -> None:
    resolved_root = root.resolve()
    env_file = resolved_root / ".env"
    if not env_file.is_file() or env_file.is_symlink():
        raise DockerPreflightError("a local regular .env file is required; copy .env.example and add secrets")
    result = run(("docker", "compose", "config", "--quiet"))
    if result.returncode != 0:
        raise DockerPreflightError("docker compose configuration is invalid; inspect the local .env values")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compose",
        action="store_true",
        help="Also require a local .env and validate docker compose config.",
    )
    arguments = parser.parse_args(argv)
    try:
        version = verify_docker_engine()
        if arguments.compose:
            verify_compose_prerequisites()
    except DockerPreflightError as error:
        print(f"Docker preflight failed: {error}", file=sys.stderr)
        return 2
    print(f"Docker Engine ready: {version}")
    if arguments.compose:
        print("Compose configuration ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
