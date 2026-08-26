"""Portable conformance checks for optional vector-index repositories."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rag_system.domain import Chunk
from rag_system.ports import IndexRepository


@dataclass(frozen=True, slots=True)
class VectorRepositoryVerification:
    """Non-sensitive evidence that a repository satisfies the base boundary."""

    index_id: str
    returned_hits: int


class VectorRepositoryConformanceError(ValueError):
    """Raised when an adapter violates the stable vector-store contract."""


def verify_index_repository(repository: IndexRepository) -> VectorRepositoryVerification:
    """Exercise build, query, lifecycle, and deletion without vendor imports.

    Run this in a disposable namespace owned by the adapter test.  It deliberately
    avoids asserting a semantic ranking: embeddings and ANN implementations may
    differ, but cross-tenant leakage, invalid scores, and unstable bounds may not.
    """

    if not all(callable(getattr(repository, name, None)) for name in ("build", "delete", "healthcheck")):
        raise TypeError("repository must implement IndexRepository")
    index_id = "contract_vector_index"
    chunks = (
        Chunk("contract-a", "document-a", "a.md", "alpha evidence", 0, 0, 14),
        Chunk("contract-b", "document-b", "b.md", "beta evidence", 0, 0, 13),
    )
    if repository.healthcheck() is not True:
        raise VectorRepositoryConformanceError("repository healthcheck is unavailable")
    index = repository.build(index_id, chunks)
    try:
        if index.index_ref.index_id != index_id or index.index_ref.chunk_count != len(chunks):
            raise VectorRepositoryConformanceError("repository returned an invalid index reference")
        hits = tuple(index.search("contract evidence", top_k=2))
        if len(hits) > 2:
            raise VectorRepositoryConformanceError("repository ignored the top_k bound")
        allowed = {chunk.chunk_id for chunk in chunks}
        previous = math.inf
        for hit in hits:
            if hit.chunk.chunk_id not in allowed or not math.isfinite(hit.score):
                raise VectorRepositoryConformanceError("repository returned an invalid search hit")
            if hit.score > previous:
                raise VectorRepositoryConformanceError("repository returned unstable score ordering")
            previous = hit.score
    finally:
        index.close()
    if repository.delete(index_id) is not True:
        raise VectorRepositoryConformanceError("repository could not delete its contract index")
    if repository.delete(index_id) is not False:
        raise VectorRepositoryConformanceError("repository deletion is not idempotent")
    return VectorRepositoryVerification(index_id=index_id, returned_hits=len(hits))


__all__ = [
    "VectorRepositoryConformanceError",
    "VectorRepositoryVerification",
    "verify_index_repository",
]
