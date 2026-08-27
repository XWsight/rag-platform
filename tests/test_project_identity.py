from __future__ import annotations

import ast
import json
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

    def test_active_package_modules_do_not_import_renamed_compatibility_modules(self) -> None:
        root = Path(__file__).resolve().parents[1]
        compatibility_modules = {
            "rag_system.assets",
            "rag_system.bootstrap",
            "rag_system.platform",
            "rag_system.service",
            "rag_system.ui",
            "rag_system.web",
        }
        compatibility_files = {f"{module.rsplit('.', maxsplit=1)[1]}.py" for module in compatibility_modules}
        violations: list[str] = []
        for source_path in sorted((root / "rag_system").glob("*.py")):
            if source_path.name in compatibility_files:
                continue
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
            imports = {
                item.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for item in node.names
            } | {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module is not None
            }
            legacy_imports = sorted(imports & compatibility_modules)
            if legacy_imports:
                violations.append(f"{source_path.name}: {', '.join(legacy_imports)}")

        self.assertEqual(violations, [])

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
