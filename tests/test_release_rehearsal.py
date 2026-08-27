from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReleaseRehearsalWorkflowTests(unittest.TestCase):
    def test_rehearsal_is_manual_and_cannot_publish_release_artifacts(self) -> None:
        workflow = (PROJECT_ROOT / ".github" / "workflows" / "release-rehearsal.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("contents: read", workflow)
        self.assertIn("push: false", workflow)
        self.assertIn("load: true", workflow)
        self.assertNotIn("gh release create", workflow)
        self.assertNotIn("packages: write", workflow)
        self.assertNotIn("attestations: write", workflow)

    def test_rehearsal_validates_the_same_release_inputs_before_building(self) -> None:
        workflow = (PROJECT_ROOT / ".github" / "workflows" / "release-rehearsal.yml").read_text(
            encoding="utf-8"
        )

        for command in (
            "python scripts/verify_dependency_lock.py",
            "python scripts/verify_openapi_contract.py",
            "python scripts/verify_wheel.py",
            "python scripts/generate_runtime_sbom.py",
            "python scripts/release_manifest.py",
        ):
            with self.subTest(command=command):
                self.assertIn(command, workflow)


if __name__ == "__main__":
    unittest.main()
