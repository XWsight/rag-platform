"""Compatibility imports; use :mod:`rag_system.rag_platform` instead."""

from rag_system.rag_platform import (
    IdempotencyInProgressError,
    KnowledgeBaseNotReadyError,
    KnowledgeBaseSubmission,
    PlatformError,
    PlatformIntegrityError,
    PlatformUnavailableError,
    PlatformValidationError,
    RagPlatform,
    UploadDocument,
)


__all__ = [
    "IdempotencyInProgressError",
    "KnowledgeBaseNotReadyError",
    "KnowledgeBaseSubmission",
    "PlatformError",
    "PlatformIntegrityError",
    "PlatformUnavailableError",
    "PlatformValidationError",
    "RagPlatform",
    "UploadDocument",
]
