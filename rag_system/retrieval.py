"""Dense, sparse, and fused retrieval implementations."""

from __future__ import annotations

import json
import math
import os
import stat
import threading
import time
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from rag_system.config import Settings
from rag_system.domain import Chunk, IndexRef, SearchHit
from rag_system.ports import Embedder, IndexRepository, Reranker, VectorIndex
from rag_system.ranking import reciprocal_rank_fusion
from rag_system.reranking import RerankerError
from rag_system.sparse import BM25Index, SparseDocument
from rag_system.text import lexical_relevance


class DependencyUnavailableError(RuntimeError):
    """Raised when an optional runtime dependency is not installed."""


class IndexIntegrityError(RuntimeError):
    """Raised when a persisted collection does not match its manifest."""


class RetrievalProfile(StrEnum):
    """Standard retrieval stages used by production and ablation runs."""

    DENSE = "dense"
    SPARSE = "sparse"
    FUSION = "fusion"
    FUSION_DIVERSE = "fusion-diverse"
    FUSION_DIVERSE_RERANK = "fusion-diverse-rerank"


@dataclass(frozen=True, slots=True)
class FusionWeights:
    """Normalized dense, BM25, lexical, and reciprocal-rank contributions."""

    dense: float = 0.55
    sparse: float = 0.0
    lexical: float = 0.25
    rrf: float = 0.20

    def __post_init__(self) -> None:
        values = (self.dense, self.sparse, self.lexical, self.rrf)
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            raise TypeError("fusion weights must be real numbers")
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("fusion weights must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("fusion weights must sum to one")

    def to_dict(self) -> dict[str, float]:
        return {
            "dense": self.dense,
            "sparse": self.sparse,
            "lexical": self.lexical,
            "rrf": self.rrf,
        }


class LocalVectorIndex:
    """A bounded in-process cosine index with no network-facing database."""

    def __init__(
        self,
        *,
        vectors: Mapping[str, tuple[float, ...]],
        index_ref: IndexRef,
        chunks: Sequence[Chunk],
        persistent: bool,
        embed_query: Callable[[str], tuple[float, ...]],
        delete_persisted: Callable[[], bool],
    ) -> None:
        self._vectors = dict(vectors)
        self._index_ref = index_ref
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._persistent = persistent
        self._embed_query = embed_query
        self._delete_persisted = delete_persisted
        self._closed = False

    @property
    def index_ref(self) -> IndexRef:
        return self._index_ref

    def search(self, query: str, *, top_k: int) -> tuple[SearchHit, ...]:
        if self._closed:
            raise RuntimeError("index is closed")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        query_vector = self._embed_query(query)
        ranked: list[tuple[str, float, float]] = []
        for chunk_id, vector in self._vectors.items():
            cosine = _cosine_similarity(query_vector, vector)
            distance = max(0.0, min(2.0, 1.0 - cosine))
            ranked.append((chunk_id, (cosine + 1.0) / 2.0, distance))
        ranked.sort(key=lambda item: (-item[1], item[0]))

        hits: list[SearchHit] = []
        for rank, (chunk_id, score, distance) in enumerate(ranked[:top_k], start=1):
            chunk = self._chunks.get(chunk_id)
            if chunk is not None:
                hits.append(
                    SearchHit(
                        chunk=chunk,
                        score=score,
                        dense_rank=rank,
                        dense_distance=distance,
                        reasons=("dense",),
                    )
                )
        return tuple(hits)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._vectors.clear()

    def delete(self) -> None:
        """Permanently delete the collection, including persisted data."""

        if self._persistent:
            self._delete_persisted()
        self._vectors.clear()
        self._closed = True


class LocalVectorIndexRepository(IndexRepository):
    """Persist a bounded local vector index using atomic, validated JSON files.

    This adapter intentionally uses exact cosine search.  It is appropriate for
    the repository's controlled, single-process deployment boundary and avoids
    exposing a vector database server.  It is not an ANN or multi-node backend.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings.validate()
        self._embedding_function: Embedder | None = None
        self._lock = threading.RLock()
        self._inference_lock = threading.RLock()
        if self.settings.persist_data:
            self._ensure_persistence_directory()

    def _embeddings(self) -> Embedder:
        with self._lock:
            if self._embedding_function is None:
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                except ImportError as error:
                    raise DependencyUnavailableError(
                        "缺少 langchain-huggingface，请先安装项目依赖。"
                    ) from error
                self._embedding_function = cast(
                    Embedder,
                    HuggingFaceEmbeddings(
                        model_name=self.settings.embedding_model,
                        encode_kwargs={"normalize_embeddings": True},
                    ),
                )
            return self._embedding_function

    def build(self, index_id: str, chunks: Sequence[Chunk]) -> LocalVectorIndex:
        if not chunks:
            raise ValueError("cannot build an empty index")
        expected_ids = tuple(chunk.chunk_id for chunk in chunks)
        if len(set(expected_ids)) != len(expected_ids):
            raise IndexIntegrityError("index contains duplicate chunk identifiers")

        vectors = self._load(index_id) if self.settings.persist_data else None
        if vectors is None or set(vectors) != set(expected_ids):
            vectors = self._embed_documents(chunks)
            if self.settings.persist_data:
                self._write(index_id, expected_ids, vectors)
        ordered_vectors = {chunk_id: vectors[chunk_id] for chunk_id in expected_ids}
        return LocalVectorIndex(
            vectors=ordered_vectors,
            index_ref=IndexRef(
                index_id=index_id,
                document_count=len({chunk.document_id for chunk in chunks}),
                chunk_count=len(chunks),
                created_at=time.time(),
            ),
            chunks=chunks,
            persistent=self.settings.persist_data,
            embed_query=self._embed_query,
            delete_persisted=lambda: self._delete(index_id),
        )

    def delete(self, index_id: str) -> bool:
        """Delete a persisted collection that is not currently in the cache."""

        if not self.settings.persist_data:
            return False
        return self._delete(index_id)

    def _delete(self, index_id: str) -> bool:
        path = self._index_path(index_id)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

    def healthcheck(self) -> bool:
        """Validate the local vector directory without opening a collection."""

        if not self.settings.persist_data:
            return True
        try:
            storage_root = self.settings.storage_root.expanduser().resolve(strict=True)
            directory = self._persistence_directory()
            metadata = directory.lstat()
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            attributes = getattr(metadata, "st_file_attributes", 0)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or bool(attributes & reparse_flag)
            ):
                return False
            directory.resolve(strict=True).relative_to(storage_root)
        except (OSError, RuntimeError, ValueError):
            return False
        return True

    def _persistence_directory(self) -> Path:
        return self.settings.storage_root.expanduser().resolve() / "vector"

    def _ensure_persistence_directory(self) -> Path:
        directory = self._persistence_directory()
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _index_path(self, index_id: str) -> Path:
        if not index_id.startswith("idx_") or not index_id[4:].isalnum():
            raise ValueError("index identifier is unsafe")
        return self._persistence_directory() / f"{index_id}.json"

    def _embed_documents(self, chunks: Sequence[Chunk]) -> dict[str, tuple[float, ...]]:
        with self._inference_lock:
            embedded = self._embeddings().embed_documents([chunk.text for chunk in chunks])
        if len(embedded) != len(chunks):
            raise IndexIntegrityError("embedding provider returned the wrong vector count")
        vectors = {
            chunk.chunk_id: _validated_vector(vector)
            for chunk, vector in zip(chunks, embedded, strict=True)
        }
        _validate_dimensions(vectors.values())
        return vectors

    def _embed_query(self, query: str) -> tuple[float, ...]:
        with self._inference_lock:
            return _validated_vector(self._embeddings().embed_query(query))

    def _load(self, index_id: str) -> dict[str, tuple[float, ...]] | None:
        path = self._index_path(index_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise IndexIntegrityError("persisted vector index is unreadable") from error
        if not isinstance(payload, dict) or set(payload) != {
            "schema",
            "embedding_model",
            "ids",
            "vectors",
        }:
            raise IndexIntegrityError("persisted vector index has an invalid schema")
        if payload["schema"] != 1 or payload["embedding_model"] != self.settings.embedding_model:
            return None
        ids, raw_vectors = payload["ids"], payload["vectors"]
        if (
            not isinstance(ids, list)
            or not all(isinstance(item, str) for item in ids)
            or len(set(ids)) != len(ids)
            or not isinstance(raw_vectors, list)
            or len(ids) != len(raw_vectors)
        ):
            raise IndexIntegrityError("persisted vector index has an invalid manifest")
        vectors = {
            item: _validated_vector(vector)
            for item, vector in zip(ids, raw_vectors, strict=True)
        }
        _validate_dimensions(vectors.values())
        return vectors

    def _write(
        self,
        index_id: str,
        ids: Sequence[str],
        vectors: Mapping[str, tuple[float, ...]],
    ) -> None:
        directory = self._ensure_persistence_directory()
        path = self._index_path(index_id)
        payload = {
            "schema": 1,
            "embedding_model": self.settings.embedding_model,
            "ids": list(ids),
            "vectors": [list(vectors[chunk_id]) for chunk_id in ids],
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{index_id}.", suffix=".tmp", dir=directory
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                os.chmod(temporary, 0o600)
                json.dump(
                    payload,
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _validated_vector(value: object) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise IndexIntegrityError("embedding provider returned an invalid vector")
    vector = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in vector):
        raise IndexIntegrityError("embedding provider returned a non-finite vector")
    return vector


def _validate_dimensions(vectors: Iterable[tuple[float, ...]]) -> None:
    dimensions = {len(vector) for vector in vectors}
    if not dimensions or len(dimensions) != 1:
        raise IndexIntegrityError("embedding vectors have inconsistent dimensions")


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise IndexIntegrityError("query and index embedding dimensions differ")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    similarity = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, similarity))


class HybridRetriever:
    """Run one explicit dense/sparse/fusion profile over a shared index."""

    def __init__(
        self,
        vector_index: VectorIndex,
        chunks: Sequence[Chunk],
        settings: Settings,
        *,
        reranker: Reranker | None = None,
        profile: RetrievalProfile | str | None = None,
        fusion_weights: FusionWeights | None = None,
    ) -> None:
        self.vector_index = vector_index
        self.settings = settings.validate()
        self.reranker = reranker
        if profile is None:
            self.profile = (
                RetrievalProfile.FUSION_DIVERSE_RERANK
                if reranker is not None
                else RetrievalProfile.FUSION_DIVERSE
            )
        else:
            try:
                self.profile = RetrievalProfile(profile)
            except (TypeError, ValueError):
                raise ValueError("invalid retrieval profile") from None
        if self.profile is RetrievalProfile.FUSION_DIVERSE_RERANK and reranker is None:
            raise ValueError("rerank profile requires a reranker")
        self.fusion_weights = fusion_weights or FusionWeights()
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._sparse = BM25Index(
            SparseDocument(document_id=chunk.chunk_id, text=chunk.text) for chunk in chunks
        )

    def search(self, query: str, *, top_k: int) -> tuple[SearchHit, ...]:
        question = (query or "").strip()
        if not question:
            return ()
        if top_k < 1:
            raise ValueError("top_k must be positive")

        if self.profile is RetrievalProfile.DENSE:
            return tuple(
                self.vector_index.search(question, top_k=self.settings.dense_candidates)
            )[:top_k]

        sparse_hits = self._sparse.search(question, top_k=self.settings.sparse_candidates)
        if self.profile is RetrievalProfile.SPARSE:
            sparse_candidates: list[SearchHit] = []
            for rank, hit in enumerate(sparse_hits, start=1):
                chunk = self._chunks.get(hit.document_id)
                if chunk is None:
                    continue
                lexical_score = lexical_relevance(question, hit.text)
                bounded_bm25_score = hit.score / (hit.score + 1.0)
                sparse_candidates.append(
                    SearchHit(
                        chunk=chunk,
                        score=bounded_bm25_score,
                        sparse_rank=rank,
                        reasons=("sparse",),
                        lexical_score=lexical_score,
                    )
                )
            return tuple(sparse_candidates[:top_k])

        dense_hits = tuple(
            self.vector_index.search(question, top_k=self.settings.dense_candidates)
        )
        dense_by_id = {hit.chunk.chunk_id: hit for hit in dense_hits}
        sparse_by_id = {hit.document_id: hit for hit in sparse_hits}
        dense_ids = [hit.chunk.chunk_id for hit in dense_hits]
        sparse_ids = [hit.document_id for hit in sparse_hits]
        fused = reciprocal_rank_fusion({"dense": dense_ids, "sparse": sparse_ids})

        maximum_rrf = 2 / 61
        candidates: list[SearchHit] = []
        for item in fused[: self.settings.fused_candidates]:
            chunk = self._chunks.get(item.item_id)
            if chunk is None:
                continue
            dense_hit = dense_by_id.get(item.item_id)
            sparse_hit = sparse_by_id.get(item.item_id)
            dense_score = dense_hit.score if dense_hit else 0.0
            sparse_score = (
                sparse_hit.score / (sparse_hit.score + 1.0)
                if sparse_hit is not None
                else 0.0
            )
            lexical_score = lexical_relevance(question, chunk.text)
            rrf_score = min(1.0, item.score / maximum_rrf)
            final_score = min(
                1.0,
                self.fusion_weights.dense * dense_score
                + self.fusion_weights.sparse * sparse_score
                + self.fusion_weights.lexical * lexical_score
                + self.fusion_weights.rrf * rrf_score,
            )
            reasons = tuple(
                reason
                for reason, present in (("dense", dense_hit is not None), ("sparse", sparse_hit is not None))
                if present
            )
            candidates.append(
                SearchHit(
                    chunk=chunk,
                    score=final_score,
                    dense_rank=dense_hit.dense_rank if dense_hit else None,
                    sparse_rank=(sparse_ids.index(item.item_id) + 1) if sparse_hit else None,
                    dense_distance=dense_hit.dense_distance if dense_hit else None,
                    reasons=reasons,
                    lexical_score=lexical_score,
                )
            )

        candidates.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        if self.profile is RetrievalProfile.FUSION_DIVERSE_RERANK:
            assert self.reranker is not None
            try:
                candidates = list(
                    self.reranker.rerank(
                        question,
                        candidates,
                        top_k=self.settings.fused_candidates,
                    )
                )
            except RerankerError:
                # Reranking is an optional quality layer; first-stage results
                # remain usable if its model is unavailable at runtime.
                pass
        if self.profile in {
            RetrievalProfile.FUSION_DIVERSE,
            RetrievalProfile.FUSION_DIVERSE_RERANK,
        }:
            candidates = self._diversify(candidates, top_k)
        return tuple(candidates[:top_k])

    @staticmethod
    def _diversify(candidates: Sequence[SearchHit], top_k: int) -> list[SearchHit]:
        selected: list[SearchHit] = []
        per_document: dict[str, int] = {}
        for hit in candidates:
            document_id = hit.chunk.document_id
            if per_document.get(document_id, 0) >= 2:
                continue
            selected.append(hit)
            per_document[document_id] = per_document.get(document_id, 0) + 1
            if len(selected) >= top_k:
                break
        return selected
