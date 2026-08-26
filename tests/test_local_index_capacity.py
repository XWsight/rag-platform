from __future__ import annotations

import unittest

from scripts.assess_local_index_capacity import assess_capacity


def _benchmark(*, p95_ms: float = 80.0) -> dict[str, object]:
    return {
        "schema_version": 3,
        "scope": "exact LocalVectorIndex.search only",
        "environment": {"source_revision": "a" * 40, "working_tree_clean": True},
        "results": [
            {"chunk_count": 1_000, "p95_ms": 20.0},
            {"chunk_count": 5_000, "p95_ms": p95_ms},
        ],
    }


class LocalIndexCapacityTests(unittest.TestCase):
    def test_measured_target_within_budget_keeps_local_exact_as_a_candidate(self) -> None:
        result = assess_capacity(_benchmark(), target_chunks=5_000, p95_budget_ms=100.0)

        self.assertEqual(result["decision"], "keep_local_exact_candidate")
        self.assertEqual(result["measured_p95_ms"], 80.0)
        self.assertIn("not an end-to-end", str(result["scope"]))

    def test_measured_target_over_budget_requests_ann_evaluation(self) -> None:
        result = assess_capacity(_benchmark(p95_ms=125.0), target_chunks=5_000, p95_budget_ms=100.0)

        self.assertEqual(result["decision"], "evaluate_ann_candidate")

    def test_unmeasured_target_requires_a_matching_measurement(self) -> None:
        result = assess_capacity(_benchmark(), target_chunks=2_000, p95_budget_ms=100.0)

        self.assertEqual(result["decision"], "measure_target")
        self.assertNotIn("measured_p95_ms", result)

    def test_duplicate_or_invalid_measurement_is_rejected(self) -> None:
        benchmark = _benchmark()
        benchmark["results"] = [
            {"chunk_count": 1_000, "p95_ms": 20.0},
            {"chunk_count": 1_000, "p95_ms": 22.0},
        ]

        with self.assertRaisesRegex(ValueError, "row is invalid"):
            assess_capacity(benchmark, target_chunks=1_000, p95_budget_ms=100.0)


if __name__ == "__main__":
    unittest.main()
