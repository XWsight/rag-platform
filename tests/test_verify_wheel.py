from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.verify_wheel import WheelVerificationError, _verify_wheel


_CONTRACT = (
    "rag-platform",
    "3.0.0.dev0",
    {
        "rag-platform-api": "rag_system.server:main",
        "rag-platform-workbench": "rag_system.workbench:main",
    },
)


def _write_wheel(
    path: Path,
    *,
    metadata: str = "Name: rag-platform\nVersion: 3.0.0.dev0\n",
    entry_points: str = (
        "[console_scripts]\n"
        "rag-platform-api = rag_system.server:main\n"
        "rag-platform-workbench = rag_system.workbench:main\n"
    ),
    retired_module: str | None = None,
) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("rag_system/__init__.py", "")
        archive.writestr("rag_system/asgi.py", "")
        if retired_module is not None:
            archive.writestr(f"rag_system/{retired_module}", "")
        archive.writestr("rag_platform-3.0.0.dist-info/METADATA", metadata)
        archive.writestr("rag_platform-3.0.0.dist-info/entry_points.txt", entry_points)


class WheelVerificationTests(unittest.TestCase):
    def test_rejects_a_wheel_with_a_retired_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel_path = Path(directory) / "invalid.whl"
            _write_wheel(wheel_path, retired_module="assets.py")
            with self.assertRaisesRegex(WheelVerificationError, "retired entrypoint"):
                _verify_wheel(wheel_path, contract=_CONTRACT)

    def test_accepts_current_package_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel_path = Path(directory) / "valid.whl"
            _write_wheel(wheel_path)
            _verify_wheel(wheel_path, contract=_CONTRACT)

    def test_rejects_mismatched_distribution_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel_path = Path(directory) / "wrong-name.whl"
            _write_wheel(
                wheel_path,
                metadata="Name: another-platform\nVersion: 3.0.0.dev0\n",
            )
            with self.assertRaisesRegex(WheelVerificationError, "distribution identity"):
                _verify_wheel(wheel_path, contract=_CONTRACT)

    def test_rejects_mismatched_console_entrypoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel_path = Path(directory) / "wrong-entrypoint.whl"
            _write_wheel(
                wheel_path,
                entry_points="[console_scripts]\nrag-platform-api = other:main\n",
            )
            with self.assertRaisesRegex(WheelVerificationError, "console entrypoints"):
                _verify_wheel(wheel_path, contract=_CONTRACT)


if __name__ == "__main__":
    unittest.main()
