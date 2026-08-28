from __future__ import annotations

import json
import unittest

from rag_system.answer_benchmark import AnswerBenchmarkMetrics, AnswerBenchmarkReport
from rag_system.application_contracts import (
    APPLICATION_CONFIGURATION_SCHEMA_VERSION,
    AnswerPolicy,
    ApplicationRevision,
    KnowledgeChatConfiguration,
)
from rag_system.application_evaluation import bind_application_evaluation


class ApplicationEvaluationTests(unittest.TestCase):
    def test_report_binds_frozen_configuration_and_benchmark_to_revision(self) -> None:
        revision = ApplicationRevision(
            revision_id="rev_0123456789abcdef0123456789abcdef",
            application_id="app_0123456789abcdef0123456789abcdef",
            revision_number=3,
            configuration_schema_version=APPLICATION_CONFIGURATION_SCHEMA_VERSION,
            configuration=KnowledgeChatConfiguration(
                knowledge_base_ids=("kb_0123456789abcdef0123456789abcdef",),
                answer_policy=AnswerPolicy(allow_cloud=True),
            ),
            created_at=10.0,
            created_by="evaluation-owner",
            change_summary="Evaluated production candidate",
        )
        benchmark = AnswerBenchmarkReport(
            dataset_digest="a" * 64, case_count=1, fact_count=1,
            metrics=AnswerBenchmarkMetrics(1.0, 1.0, 1.0, 1.0, 1.0), results=(),
        )

        report = bind_application_evaluation(revision, benchmark, generated_at=11.0)
        payload = json.loads(report.to_json())

        self.assertEqual(payload["application_id"], revision.application_id)
        self.assertEqual(payload["revision_id"], revision.revision_id)
        self.assertEqual(payload["benchmark"]["dataset_digest"], "a" * 64)
        self.assertEqual(len(payload["configuration_digest"]), 64)
        self.assertNotIn("configuration", payload)


if __name__ == "__main__":
    unittest.main()
