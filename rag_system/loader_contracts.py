"""Public loading limits, supported formats, and safe error classifications."""

from __future__ import annotations

from dataclasses import dataclass

from rag_system.security import DocumentValidationError


SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".html", ".htm", ".docx", ".pdf"})
TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown"})
HTML_EXTENSIONS = frozenset({".html", ".htm"})


class DocumentLoadError(DocumentValidationError):
    """A document could not be loaded within the configured safety boundary."""


class MissingDocumentDependencyError(DocumentLoadError):
    """A selected format needs an optional parser that is not installed."""


@dataclass(frozen=True, slots=True)
class LoaderLimits:
    """Hard resource limits applied before and during document parsing."""

    max_documents: int = 10
    max_file_bytes: int = 5 * 1024 * 1024
    max_total_file_bytes: int = 20 * 1024 * 1024
    max_uncompressed_bytes: int = 20 * 1024 * 1024
    max_archive_members: int = 512
    max_compression_ratio: float = 200.0
    max_pages: int = 200
    max_paragraphs: int = 20_000
    max_characters: int = 2_000_000

    def validate(self) -> LoaderLimits:
        integer_limits = (
            self.max_documents,
            self.max_file_bytes,
            self.max_total_file_bytes,
            self.max_uncompressed_bytes,
            self.max_archive_members,
            self.max_pages,
            self.max_paragraphs,
            self.max_characters,
        )
        if any(value < 1 for value in integer_limits):
            raise ValueError("all loader limits must be positive")
        if self.max_total_file_bytes < self.max_file_bytes:
            raise ValueError("max_total_file_bytes cannot be smaller than max_file_bytes")
        if self.max_compression_ratio < 1.0:
            raise ValueError("max_compression_ratio must be at least 1")
        return self


__all__ = [
    "DocumentLoadError",
    "HTML_EXTENSIONS",
    "LoaderLimits",
    "MissingDocumentDependencyError",
    "SUPPORTED_EXTENSIONS",
    "TEXT_EXTENSIONS",
]
