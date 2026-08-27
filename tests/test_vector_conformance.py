from __future__ import annotations

import unittest

from rag_system.domain import IndexRef, SearchHit
from rag_system.vector_conformance import (
    VectorRepositoryConformanceError,
    verify_index_repository,
)


class _Index:
    def __init__(self, index_id, chunks) -> None:
        self.index_ref = IndexRef(index_id, 2, len(chunks), 1.0)
        self.chunks = tuple(chunks)

    def search(self, _query, *, top_k):
        return tuple(SearchHit(chunk, 1.0 - index) for index, chunk in enumerate(self.chunks[:top_k]))

    def close(self) -> None:
        return None

    def delete(self) -> None:
        return None


class _Repository:
    def __init__(self) -> None:
        self.indexes = {}

    def healthcheck(self):
        return True

    def build(self, index_id, chunks):
        index = _Index(index_id, chunks)
        self.indexes[index_id] = index
        return index

    def delete(self, index_id):
        return self.indexes.pop(index_id, None) is not None


class VectorConformanceTests(unittest.TestCase):
    def test_repository_contract_is_vendor_neutral_and_executable(self) -> None:
        result = verify_index_repository(_Repository())
        self.assertEqual(result.index_id, "contract_vector_index")
        self.assertEqual(result.returned_hits, 2)

    def test_non_idempotent_delete_is_rejected(self) -> None:
        class BadRepository(_Repository):
            def delete(self, _index_id):
                return True

        with self.assertRaises(VectorRepositoryConformanceError):
            verify_index_repository(BadRepository())

    def test_cross_index_result_leakage_is_rejected(self) -> None:
        class LeakyRepository(_Repository):
            def __init__(self) -> None:
                super().__init__()
                self.first_chunks = ()

            def build(self, index_id, chunks):
                if not self.first_chunks:
                    self.first_chunks = tuple(chunks)
                return super().build(index_id, self.first_chunks)

        with self.assertRaisesRegex(VectorRepositoryConformanceError, "search hit"):
            verify_index_repository(LeakyRepository())


if __name__ == "__main__":
    unittest.main()
