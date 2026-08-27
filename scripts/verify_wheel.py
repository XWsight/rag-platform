"""Build a clean wheel and reject retired entrypoint modules from its contents."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_FILES = ("pyproject.toml", "README.md", "LICENSE")
_RETIRED_MODULES = frozenset({"assets.py", "bootstrap.py", "platform.py", "service.py", "ui.py", "web.py"})


class WheelVerificationError(RuntimeError):
    """The generated distribution does not match the current public package contract."""


def build_and_verify(*, project_root: Path = PROJECT_ROOT) -> Path:
    """Build from a copied source tree so an ignored local ``build/`` cannot leak files."""

    root = project_root.resolve()
    with tempfile.TemporaryDirectory(prefix="rag-platform-wheel-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        source_root = temporary_root / "source"
        wheel_root = temporary_root / "wheel"
        source_root.mkdir()
        wheel_root.mkdir()
        for filename in _SOURCE_FILES:
            shutil.copy2(root / filename, source_root / filename)
        shutil.copytree(root / "rag_system", source_root / "rag_system", ignore=_ignore_cache_files)
        _run_wheel_build(source_root, wheel_root)
        wheel_files = tuple(wheel_root.glob("*.whl"))
        if len(wheel_files) != 1:
            raise WheelVerificationError("wheel build did not produce exactly one artifact")
        _verify_wheel(wheel_files[0])
        destination = root / "dist" / wheel_files[0].name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(wheel_files[0], destination)
        return destination


def _ignore_cache_files(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name == "__pycache__" or name.endswith(".pyc")}


def _run_wheel_build(source_root: Path, wheel_root: Path) -> None:
    result = subprocess.run(
        (sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-cache-dir", "--wheel-dir", str(wheel_root), str(source_root)),
        check=False,
        capture_output=True,
        encoding="utf-8",
        timeout=180,
    )
    if result.returncode != 0:
        raise WheelVerificationError("clean wheel build failed")


def _verify_wheel(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as archive:
        names = frozenset(archive.namelist())
    if "rag_system/__init__.py" not in names or "rag_system/asgi.py" not in names:
        raise WheelVerificationError("wheel is missing current package entrypoints")
    leaked = sorted(name for name in names if Path(name).name in _RETIRED_MODULES)
    if leaked:
        raise WheelVerificationError("wheel contains retired entrypoint modules")
    if not any(name.endswith(".dist-info/METADATA") for name in names):
        raise WheelVerificationError("wheel is missing distribution metadata")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional verified wheel destination.")
    arguments = parser.parse_args(argv)
    try:
        wheel_path = build_and_verify()
        if arguments.output is not None:
            destination = arguments.output.resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wheel_path, destination)
            wheel_path = destination
    except (OSError, subprocess.SubprocessError, WheelVerificationError) as error:
        print(f"wheel verification rejected: {error}", file=sys.stderr)
        return 2
    print(f"verified wheel: {wheel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
