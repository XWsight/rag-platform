from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from rag_system.provenance import SourceProvenance
from scripts.release_manifest import release_manifest, require_clean


class ReleaseManifestTests(unittest.TestCase):
    def test_manifest_hashes_only_declared_build_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {}
            for relative in (
                "Dockerfile",
                "compose.yaml",
                "pyproject.toml",
                "requirements.txt",
                "requirements-dev.txt",
            ):
                content = b'[project]\nversion = "2.0.0"\n' if relative == "pyproject.toml" else relative.encode()
                (root / relative).write_bytes(content)
                expected[relative] = hashlib.sha256(content).hexdigest()
            (root / ".env").write_text("SUPER_SECRET=must-not-be-read", encoding="utf-8")

            with patch(
                "scripts.release_manifest.inspect_source_provenance",
                return_value=SourceProvenance(
                    revision="a" * 40,
                    working_tree_clean=True,
                ),
            ):
                manifest = release_manifest(
                    root=root,
                    generated_at=datetime(2026, 8, 26, tzinfo=UTC),
                )

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["source_revision"], "a" * 40)
        self.assertTrue(manifest["working_tree_clean"])
        self.assertEqual(manifest["package_version"], "2.0.0")
        self.assertEqual(manifest["build_inputs"], expected)
        self.assertNotIn(".env", manifest["build_inputs"])
        self.assertEqual(manifest["generated_at"], "2026-08-26T00:00:00+00:00")

    def test_release_requires_a_clean_versioned_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "source revision"):
            require_clean({"source_revision": None, "working_tree_clean": True})
        with self.assertRaisesRegex(ValueError, "clean"):
            require_clean({"source_revision": "a" * 40, "working_tree_clean": False})


if __name__ == "__main__":
    unittest.main()
