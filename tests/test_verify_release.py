from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.verify_release import package_version, verify_release_tag


class VerifyReleaseTests(unittest.TestCase):
    def test_reads_a_stable_semantic_package_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text('[project]\nversion = "2.1.0"\n', encoding="utf-8")
            version = package_version(root=root)

        self.assertEqual(version, "2.1.0")
        verify_release_tag("v2.1.0", version=version)

    def test_rejects_development_version_and_mismatched_tag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text('[project]\nversion = "2.1.0.dev0"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stable semantic"):
                package_version(root=root)

        with self.assertRaisesRegex(ValueError, "exactly match"):
            verify_release_tag("v2.1.1", version="2.1.0")


if __name__ == "__main__":
    unittest.main()
