"""Build a clean wheel and reject retired entrypoint modules from its contents."""

from __future__ import annotations

import argparse
import configparser
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_FILES = ("pyproject.toml", "README.md", "LICENSE")
_RETIRED_MODULES = frozenset(
    {"assets.py", "bootstrap.py", "platform.py", "service.py", "ui.py", "web.py"}
)


class WheelVerificationError(RuntimeError):
    """The generated distribution does not match the current public package contract."""


def build_and_verify(
    *,
    project_root: Path = PROJECT_ROOT,
    output_path: Path | None = None,
) -> Path:
    """Build from a copied source tree so an ignored local ``build/`` cannot leak files."""

    root = project_root.resolve()
    contract = _load_distribution_contract(root)
    with tempfile.TemporaryDirectory(prefix="rag-platform-wheel-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        source_root = temporary_root / "source"
        wheel_root = temporary_root / "wheel"
        source_root.mkdir()
        wheel_root.mkdir()
        for filename in _SOURCE_FILES:
            shutil.copy2(root / filename, source_root / filename)
        shutil.copytree(
            root / "rag_system",
            source_root / "rag_system",
            ignore=_ignore_cache_files,
        )
        _run_wheel_build(source_root, wheel_root)
        wheel_files = tuple(wheel_root.glob("*.whl"))
        if len(wheel_files) != 1:
            raise WheelVerificationError("wheel build did not produce exactly one artifact")
        _verify_wheel(wheel_files[0], contract=contract)
        if output_path is None:
            return Path(wheel_files[0].name)
        destination = output_path.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wheel_files[0], destination)
        return destination


def _ignore_cache_files(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name == "__pycache__" or name.endswith(".pyc")}


def _run_wheel_build(source_root: Path, wheel_root: Path) -> None:
    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-cache-dir",
            "--wheel-dir",
            str(wheel_root),
            str(source_root),
        ),
        check=False,
        capture_output=True,
        encoding="utf-8",
        timeout=180,
    )
    if result.returncode != 0:
        raise WheelVerificationError("clean wheel build failed")


def _load_distribution_contract(root: Path) -> tuple[str, str, dict[str, str]]:
    try:
        payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        project = payload["project"]
        name = project["name"]
        version = project["version"]
        scripts = project.get("scripts", {})
    except (KeyError, OSError, tomllib.TOMLDecodeError, TypeError) as error:
        raise WheelVerificationError("project distribution metadata is invalid") from error
    if (
        not isinstance(name, str)
        or not isinstance(version, str)
        or not isinstance(scripts, dict)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in scripts.items())
    ):
        raise WheelVerificationError("project distribution metadata is invalid")
    return name, version, dict(scripts)


def _verify_wheel(
    wheel_path: Path,
    *,
    contract: tuple[str, str, dict[str, str]],
) -> None:
    with zipfile.ZipFile(wheel_path) as archive:
        names = frozenset(archive.namelist())
        metadata_paths = tuple(name for name in names if name.endswith(".dist-info/METADATA"))
        entry_point_paths = tuple(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        if len(metadata_paths) != 1 or len(entry_point_paths) != 1:
            raise WheelVerificationError("wheel has incomplete distribution metadata")
        metadata = BytesParser(policy=email_policy).parsebytes(archive.read(metadata_paths[0]))
        entry_points = _parse_entry_points(archive.read(entry_point_paths[0]))
    if "rag_system/__init__.py" not in names or "rag_system/asgi.py" not in names:
        raise WheelVerificationError("wheel is missing current package entrypoints")
    leaked = sorted(name for name in names if Path(name).name in _RETIRED_MODULES)
    if leaked:
        raise WheelVerificationError("wheel contains retired entrypoint modules")
    name, version, scripts = contract
    if metadata.get("Name") != name or metadata.get("Version") != version:
        raise WheelVerificationError("wheel distribution identity does not match pyproject.toml")
    if entry_points != scripts:
        raise WheelVerificationError("wheel console entrypoints do not match pyproject.toml")


def _parse_entry_points(raw: bytes) -> dict[str, str]:
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(raw.decode("utf-8"))
    except (UnicodeDecodeError, configparser.Error) as error:
        raise WheelVerificationError("wheel console entrypoints are invalid") from error
    if set(parser.sections()) != {"console_scripts"}:
        raise WheelVerificationError("wheel console entrypoints are invalid")
    return dict(parser["console_scripts"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional verified wheel destination.")
    arguments = parser.parse_args(argv)
    try:
        wheel_path = build_and_verify(output_path=arguments.output)
    except (OSError, subprocess.SubprocessError, WheelVerificationError) as error:
        print(f"wheel verification rejected: {error}", file=sys.stderr)
        return 2
    print(f"verified wheel: {wheel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
