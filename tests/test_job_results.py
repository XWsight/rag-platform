from __future__ import annotations

import unittest

from rag_system.job_results import InvalidJobResultError, canonical_job_result


class JobResultTests(unittest.TestCase):
    def test_canonical_result_is_deterministic_and_bounded(self) -> None:
        value = {"b": [True, 2], "a": {"nested": None}}
        self.assertEqual(
            canonical_job_result(value, max_bytes=100, max_depth=4, max_items=10),
            '{"a":{"nested":null},"b":[true,2]}',
        )

    def test_result_rejects_cycles_non_finite_values_and_non_object_roots(self) -> None:
        cyclic: list[object] = []
        cyclic.append(cyclic)
        for value in (cyclic, {"nan": float("nan")}, ["not-an-object"]):
            with self.subTest(value=type(value).__name__), self.assertRaises(InvalidJobResultError):
                canonical_job_result(value, max_bytes=100, max_depth=4, max_items=10)


if __name__ == "__main__":
    unittest.main()
