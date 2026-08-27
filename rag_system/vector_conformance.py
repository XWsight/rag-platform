"""Portable conformance checks for optional vector-index repositories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from rag_system.domain import Chunk, SearchHit
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
    secondary_index_id = "contract_vector_index_secondary"
    chunks = (
        Chunk("contract-a", "document-a", "a.md", "alpha evidence", 0, 0, 14),
        Chunk("contract-b", "document-b", "b.md", "beta evidence", 0, 0, 13),
    )
    secondary_chunks = (
        Chunk("contract-c", "document-c", "c.md", "gamma evidence", 0, 0, 14),
        Chunk("contract-d", "document-d", "d.md", "delta evidence", 0, 0, 14),
    )
    if repository.healthcheck() is not True:
        raise VectorRepositoryConformanceError("repository healthcheck is unavailable")
    index = repository.build(index_id, chunks)
    secondary = repository.build(secondary_index_id, secondary_chunks)
    try:
        hits = _verify_index(index, index_id=index_id, chunks=chunks)
        _verify_index(secondary, index_id=secondary_index_id, chunks=secondary_chunks)
    finally:
        index.close()
        secondary.close()
    if repository.delete(index_id) is not True:
        raise VectorRepositoryConformanceError("repository could not delete its contract index")
    if repository.delete(index_id) is not False:
        raise VectorRepositoryConformanceError("repository deletion is not idempotent")
    # A delete must affect only its exact collection. This is the minimum
    # tenant/namespace isolation property a generic repository can prove
    # without knowing a vendor's connection or credential model.
    surviving = repository.build(secondary_index_id, secondary_chunks)
    try:
        _verify_index(surviving, index_id=secondary_index_id, chunks=secondary_chunks)
    finally:
        surviving.close()
    if repository.delete(secondary_index_id) is not True:
        raise VectorRepositoryConformanceError("repository could not delete its secondary contract index")
    return VectorRepositoryVerification(index_id=index_id, returned_hits=len(hits))


def _verify_index(
    index: object,
    *,
    index_id: str,
    chunks: tuple[Chunk, ...],
) -> tuple[SearchHit, ...]:
    """Validate one isolated collection without assuming a ranking algorithm."""

    index_ref = getattr(index, "index_ref", None)
    search = getattr(index, "search", None)
    if index_ref is None or not callable(search):
        raise VectorRepositoryConformanceError("repository returned an invalid vector index")
    if index_ref.index_id != index_id or index_ref.chunk_count != len(chunks):
        raise VectorRepositoryConformanceError("repository returned an invalid index reference")
    hits = cast(tuple[SearchHit, ...], tuple(search("contract evidence", top_k=2)))
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
    return hits


__all__ = [
    "VectorRepositoryConformanceError",
    "VectorRepositoryVerification",
    "verify_index_repository",
]
