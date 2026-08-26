from __future__ import annotations

import unittest

from scripts.benchmark_local_index import build_synthetic_index, parse_sizes, percentile, run
from rag_system.provenance import SourceProvenance


class LocalIndexBenchmarkTests(unittest.TestCase):
    def test_parse_sizes_rejects_duplicates(self) -> None:
        with self.assertRaisesRegex(Exception, "duplicates"):
            parse_sizes("10,10")

    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertEqual(percentile([10, 20, 30, 40], 0.50), 25)
        self.assertEqual(percentile([10, 20, 30, 40], 0.95), 38)

    def test_synthetic_index_returns_requested_hits(self) -> None:
        index = build_synthetic_index(8, 4)
        try:
            self.assertEqual(len(index.search("query", top_k=3)), 3)
        finally:
            index.close()

    def test_report_states_its_scope_and_each_size(self) -> None:
        report = run(
            sizes=(5, 8),
            dimension=4,
            queries=2,
            warmup=1,
            top_k=2,
            provenance=SourceProvenance(
                revision="a" * 40,
                working_tree_clean=True,
            ),
        )
        self.assertEqual(report["schema_version"], 3)
        self.assertIn("excludes embedding", str(report["scope"]))
        environment = report["environment"]
        self.assertIsInstance(environment, dict)
        self.assertTrue(str(environment["python_version"]))
        self.assertTrue(str(environment["platform"]))
        self.assertTrue(
            environment["cpu_count"] is None or isinstance(environment["cpu_count"], int)
        )
        self.assertEqual(environment["source_revision"], "a" * 40)
        self.assertIs(environment["working_tree_clean"], True)
        configuration = report["configuration"]
        self.assertEqual(configuration["sizes"], [5, 8])
        self.assertEqual(configuration["dimension"], 4)
        rows = report["results"]
        self.assertIsInstance(rows, list)
        self.assertEqual([row["chunk_count"] for row in rows], [5, 8])
        self.assertTrue(all(row["p95_ms"] >= row["min_ms"] for row in rows))


if __name__ == "__main__":
    unittest.main()
