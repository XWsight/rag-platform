from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rag_system.config import Settings
from rag_system.domain import Chunk
from rag_system.retrieval import IndexIntegrityError, LocalVectorIndexRepository


class FakeEmbedder:
    def __init__(self) -> None:
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [[float(len(text)), float(index + 1)] for index, text in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


def make_chunk(chunk_id: str, text: str | None = None) -> Chunk:
    payload = text or f"content for {chunk_id}"
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_name="guide.txt",
        text=payload,
        chunk_index=0,
        start_char=0,
        end_char=len(payload),
    )


class LocalVectorLifecycleTests(unittest.TestCase):
    @staticmethod
    def _repository(
        directory: str, *, persistent: bool
    ) -> tuple[LocalVectorIndexRepository, FakeEmbedder]:
        settings = replace(
            Settings(),
            persist_data=persistent,
            storage_root=Path(directory),
        ).validate()
        repository = LocalVectorIndexRepository(settings)
        embedder = FakeEmbedder()
        repository._embedding_function = embedder
        return repository, embedder

    def test_persisted_index_reopens_without_reembedding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, embedder = self._repository(directory, persistent=True)
            chunks = (make_chunk("chunk1"), make_chunk("chunk2"))
            first = repository.build("idx_stable", chunks)
            first.close()
            second = repository.build("idx_stable", chunks)

            self.assertEqual(embedder.document_calls, 1)
            self.assertEqual(len(second.search("content", top_k=2)), 2)
            second.delete()
            self.assertFalse((Path(directory) / "vector" / "idx_stable.json").exists())

    def test_changed_manifest_rebuilds_and_replaces_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, embedder = self._repository(directory, persistent=True)
            repository.build("idx_stable", (make_chunk("chunk1"),)).close()
            repository.build("idx_stable", (make_chunk("chunk2"),)).close()

            self.assertEqual(embedder.document_calls, 2)

    def test_ephemeral_index_does_not_write_a_persistence_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, _ = self._repository(directory, persistent=False)
            index = repository.build("idx_ephemeral", (make_chunk("chunk1"),))
            index.close()
            self.assertFalse((Path(directory) / "vector").exists())

    def test_corrupted_persisted_index_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, _ = self._repository(directory, persistent=True)
            path = Path(directory) / "vector" / "idx_stable.json"
            path.write_text("{not-json", encoding="utf-8")
            with self.assertRaises(IndexIntegrityError):
                repository.build("idx_stable", (make_chunk("chunk1"),))


if __name__ == "__main__":
    unittest.main()
