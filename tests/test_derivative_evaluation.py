from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.validate_derivative_evaluation import (
    DerivativeEvaluationError,
    validate_governance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DerivativeEvaluationGovernanceTests(unittest.TestCase):
    def test_draft_governance_is_explicitly_non_release_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "governance.json"
            path.write_text(json.dumps(_governance(status="draft")), encoding="utf-8")

            self.assertEqual(validate_governance(path)["status"], "draft")
            with self.assertRaisesRegex(DerivativeEvaluationError, "still draft"):
                validate_governance(path, require_ready=True)

    def test_ready_governance_validates_frozen_suites_and_held_out_test_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evaluation_root = Path(directory) / "evals"
            shutil.copytree(PROJECT_ROOT / "evals", evaluation_root)
            path = evaluation_root / "governance.json"
            path.write_text(json.dumps(_governance(status="ready")), encoding="utf-8")

            result = validate_governance(path, require_ready=True)

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["retrieval_cases"], 216)
            self.assertEqual(result["answer_cases"], 50)

    def test_consumed_held_out_data_blocks_release_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "governance.json"
            path.write_text(
                json.dumps(_governance(status="ready", held_out_test_status="consumed")),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DerivativeEvaluationError, "already been consumed"):
                validate_governance(path, require_ready=True)

    def test_template_owner_cannot_be_marked_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "governance.json"
            governance = _governance(status="ready")
            governance["owner"] = "replace-with-domain-evaluation-owner"
            path.write_text(json.dumps(governance), encoding="utf-8")

            with self.assertRaisesRegex(DerivativeEvaluationError, "template placeholder"):
                validate_governance(path, require_ready=True)


def _governance(
    *,
    status: str,
    held_out_test_status: str = "unconsumed",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "product_name": "Derivative Assistant",
        "base_revision": "0123456789abcdef",
        "owner": "evaluation-owner",
        "data_classification": "internal",
        "status": status,
        "held_out_test_status": held_out_test_status,
        "retrieval": {
            "suite": "retrieval_suite.json",
            "contract": "gates/retrieval-suite.json",
        },
        "answer": {
            "suite": "answer_suite.json",
            "contract": "gates/answer-suite.json",
        },
    }


if __name__ == "__main__":
    unittest.main()
