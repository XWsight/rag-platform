from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.verify_wheel import WheelVerificationError, _verify_wheel


class WheelVerificationTests(unittest.TestCase):
    def test_rejects_a_wheel_with_a_retired_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel_path = Path(directory) / "invalid.whl"
            with zipfile.ZipFile(wheel_path, "w") as archive:
                archive.writestr("rag_system/__init__.py", "")
                archive.writestr("rag_system/asgi.py", "")
                archive.writestr("rag_system/assets.py", "")
                archive.writestr("rag_platform-3.0.0.dist-info/METADATA", "")
            with self.assertRaisesRegex(WheelVerificationError, "retired entrypoint"):
                _verify_wheel(wheel_path)

    def test_accepts_current_package_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel_path = Path(directory) / "valid.whl"
            with zipfile.ZipFile(wheel_path, "w") as archive:
                archive.writestr("rag_system/__init__.py", "")
                archive.writestr("rag_system/asgi.py", "")
                archive.writestr("rag_platform-3.0.0.dist-info/METADATA", "")
            _verify_wheel(wheel_path)


if __name__ == "__main__":
    unittest.main()
