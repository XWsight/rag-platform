from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SecretSourceWorkflowTests(unittest.TestCase):
    def test_quality_workflow_exercises_real_compose_secret_mounts(self) -> None:
        workflow = (PROJECT_ROOT / ".github" / "workflows" / "quality.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("secret-source-smoke:", workflow)
        self.assertIn("compose.secrets.example.yaml", workflow)
        self.assertIn("compose-secret-key-0123456789abcdef", workflow)
        self.assertIn("plaintext-key-must-not-work-0123456789", workflow)
        self.assertIn('test "$status" = "401"', workflow)
        self.assertIn("- secret-source-smoke", workflow)


if __name__ == "__main__":
    unittest.main()
