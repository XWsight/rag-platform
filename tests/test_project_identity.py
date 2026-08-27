from __future__ import annotations

import json
import re
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

    def test_frontend_tooling_uses_the_current_product_identity(self) -> None:
        root = Path(__file__).resolve().parents[1]
        frontend = json.loads((root / "package.json").read_text(encoding="utf-8"))
        lock = json.loads((root / "package-lock.json").read_text(encoding="utf-8"))

        self.assertEqual(frontend["name"], "rag-platform")
        self.assertEqual(lock["name"], "rag-platform")
        self.assertEqual(lock["packages"][""]["name"], "rag-platform")

    def test_package_does_not_ship_retired_entrypoint_modules(self) -> None:
        root = Path(__file__).resolve().parents[1]
        retired_modules = {"assets.py", "bootstrap.py", "platform.py", "service.py", "ui.py", "web.py"}
        self.assertEqual(retired_modules & {path.name for path in (root / "rag_system").glob("*.py")}, set())

    def test_public_assets_do_not_reference_retired_v3_module_names(self) -> None:
        root = Path(__file__).resolve().parents[1]
        retired_modules = {"assets.py", "bootstrap.py", "platform.py", "service.py", "ui.py", "web.py"}
        retired_pattern = re.compile(
            r"(?<![A-Za-z0-9_])(?:"
            + "|".join(re.escape(name) for name in retired_modules)
            + r")(?![A-Za-z0-9_])"
        )
        candidates = [root / name for name in ("README.md", "CHANGELOG.md", "CONTRIBUTING.md", "RELEASE.md")]
        for directory in (root / ".github", root / "docs", root / "templates"):
            candidates.extend(
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix in {".json", ".md", ".template", ".yaml", ".yml"}
            )

        violations = [
            str(path.relative_to(root))
            for path in candidates
            if path.is_file()
            and retired_pattern.search(path.read_text(encoding="utf-8")) is not None
        ]
        self.assertEqual(violations, [])

    def test_source_and_public_assets_do_not_reference_the_retired_product_identity(self) -> None:
        root = Path(__file__).resolve().parents[1]
        retired_identities = ("rag" + "-studio", "rag" + " studio")
        candidates = [
            path
            for directory in (
                root / ".github",
                root / "docs",
                root / "rag_system",
                root / "scripts",
                root / "templates",
                root / "tests",
            )
            for path in directory.rglob("*")
            if path.is_file() and path.suffix in {".json", ".js", ".md", ".mjs", ".py", ".template", ".toml", ".yaml", ".yml"}
        ]
        candidates.extend(path for path in (root / "compose.yaml", root / "README.md", root / ".env.example") if path.is_file())
        self.assertEqual(
            [
                str(path.relative_to(root))
                for path in candidates
                if any(identity in path.read_text(encoding="utf-8").casefold() for identity in retired_identities)
            ],
            [],
        )

    def test_public_project_assets_do_not_reference_the_retired_distribution_name(self) -> None:
        root = Path(__file__).resolve().parents[1]
        retired_name = "rag" + "-system"
        candidates = [
            root / ".env.example",
            root / "CHANGELOG.md",
            root / "Dockerfile",
            root / "README.md",
            root / "compose.yaml",
            root / "package.json",
            root / "package-lock.json",
            root / "pyproject.toml",
        ]
        for directory in (root / ".github", root / "docs", root / "scripts", root / "templates"):
            candidates.extend(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and path.suffix
                in {".json", ".js", ".md", ".mjs", ".ps1", ".py", ".template", ".toml", ".yaml", ".yml"}
            )

        violations = [
            str(path.relative_to(root))
            for path in candidates
            if retired_name in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(violations, [])

if __name__ == "__main__":
    unittest.main()
