from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from rag_system import __version__


class ProjectIdentityTests(unittest.TestCase):
    def test_runtime_version_matches_package_metadata(self) -> None:
        root = Path(__file__).resolve().parents[1]
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(project["project"]["name"], "rag-platform")
        self.assertEqual(project["project"]["version"], __version__)
        self.assertEqual(
            project["project"]["scripts"],
            {
                "rag-platform-api": "rag_system.server:main",
                "rag-platform-workbench": "rag_system.workbench:main",
            },
        )

    def test_package_declares_typing_and_public_project_metadata(self) -> None:
        root = Path(__file__).resolve().parents[1]
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(project["project"]["license"], "MIT")
        self.assertIn("Homepage", project["project"]["urls"])
        self.assertIn(
            "py.typed",
            project["tool"]["setuptools"]["package-data"]["rag_system"],
        )
        self.assertTrue((root / "rag_system" / "py.typed").is_file())

if __name__ == "__main__":
    unittest.main()
