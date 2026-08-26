"""Storage-neutral knowledge-base records, states, and validation rules.

These values form the contract shared by use cases, delivery adapters, and
repository implementations.  They deliberately do not depend on SQLite or a
specific catalog implementation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from rag_system.tenancy import TenantId


MAX_DOCUMENT_MANIFEST_ITEMS = 10_000

_RESOURCE_ID_PATTERN = re.compile(r"kb_[A-Za-z0-9_-]{32}")
_INDEX_ID_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._:-]{0,190}[A-Za-z0-9])?")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_IDEMPOTENCY_RESERVATION_PATTERN = re.compile(r"idem_[0-9a-f]{32}")


class KnowledgeBaseContractError(Exception):
    """Base error for invalid storage-neutral knowledge-base values."""


class KnowledgeBaseValidationError(KnowledgeBaseContractError, ValueError):
    """A knowledge-base record or state value violates its contract."""


class KnowledgeBaseStatus(StrEnum):
    PREPARING = "preparing"
    PENDING = "pending"
    INDEXING = "indexing"
    CANCELLING = "cancelling"
    READY = "ready"
    FAILED = "failed"
    DELETING = "deleting"


class KnowledgeBaseErrorCode(StrEnum):
    CONTENT_REJECTED = "content_rejected"
    INGESTION_FAILED = "ingestion_failed"
    INDEX_BUILD_FAILED = "index_build_failed"
    INDEX_STORAGE_FAILED = "index_storage_failed"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    INTERNAL_ERROR = "internal_error"
    INDEX_CANCELLED = "index_cancelled"


ALLOWED_STATUS_TRANSITIONS: Mapping[KnowledgeBaseStatus, frozenset[KnowledgeBaseStatus]] = {
    KnowledgeBaseStatus.PREPARING: frozenset(
        {
            KnowledgeBaseStatus.PENDING,
            KnowledgeBaseStatus.FAILED,
            KnowledgeBaseStatus.DELETING,
        }
    ),
    KnowledgeBaseStatus.PENDING: frozenset(
        {
            KnowledgeBaseStatus.INDEXING,
            KnowledgeBaseStatus.CANCELLING,
            KnowledgeBaseStatus.FAILED,
            KnowledgeBaseStatus.DELETING,
        }
    ),
    KnowledgeBaseStatus.INDEXING: frozenset(
        {
            KnowledgeBaseStatus.READY,
            KnowledgeBaseStatus.CANCELLING,
            KnowledgeBaseStatus.FAILED,
            KnowledgeBaseStatus.DELETING,
        }
    ),
    KnowledgeBaseStatus.READY: frozenset(
        {KnowledgeBaseStatus.INDEXING, KnowledgeBaseStatus.DELETING}
    ),
    KnowledgeBaseStatus.CANCELLING: frozenset(
        {KnowledgeBaseStatus.FAILED, KnowledgeBaseStatus.DELETING}
    ),
    KnowledgeBaseStatus.FAILED: frozenset(
        {KnowledgeBaseStatus.INDEXING, KnowledgeBaseStatus.DELETING}
    ),
    KnowledgeBaseStatus.DELETING: frozenset(),
}


@dataclass(frozen=True, slots=True)
class DocumentManifest:
    display_name: str
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "display_name", validate_display_name(self.display_name, 255))
        object.__setattr__(self, "relative_path", validate_relative_path(self.relative_path))
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise KnowledgeBaseValidationError("Document size must be an integer.")
        if not 0 <= self.size_bytes <= 2**63 - 1:
            raise KnowledgeBaseValidationError("Document size is outside the supported range.")
        if not isinstance(self.sha256, str) or _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise KnowledgeBaseValidationError(
                "Document SHA-256 must be 64 lowercase hexadecimal characters."
            )


DocumentManifestEntry = DocumentManifest


@dataclass(frozen=True, slots=True)
class KnowledgeBaseRecord:
    resource_id: str
    tenant_id: TenantId
    display_name: str
    status: KnowledgeBaseStatus
    internal_index_id: str | None
    documents: tuple[DocumentManifest, ...]
    document_count: int
    total_bytes: int
    chunk_count: int
    error_code: KnowledgeBaseErrorCode | None
    created_at: float
    updated_at: float
    version: int
    idempotency_reservation_id: str | None = None

    def __post_init__(self) -> None:
        validate_resource_id(self.resource_id)
        if not isinstance(self.tenant_id, TenantId):
            raise KnowledgeBaseValidationError("tenant_id must be a TenantId.")
        object.__setattr__(self, "display_name", validate_display_name(self.display_name, 200))
        if not isinstance(self.status, KnowledgeBaseStatus):
            raise KnowledgeBaseValidationError("Invalid knowledge base status.")
        if self.internal_index_id is not None:
            object.__setattr__(
                self,
                "internal_index_id",
                validate_internal_index_id(self.internal_index_id),
            )
        object.__setattr__(self, "documents", normalize_manifest(self.documents))
        expected_total = sum(item.size_bytes for item in self.documents)
        if self.document_count != len(self.documents) or self.total_bytes != expected_total:
            raise KnowledgeBaseValidationError("Stored document counts do not match the manifest.")
        if isinstance(self.chunk_count, bool) or not isinstance(self.chunk_count, int) or self.chunk_count < 0:
            raise KnowledgeBaseValidationError("chunk_count must be a non-negative integer.")
        if self.status in {
            KnowledgeBaseStatus.PENDING,
            KnowledgeBaseStatus.INDEXING,
            KnowledgeBaseStatus.CANCELLING,
            KnowledgeBaseStatus.READY,
        } and not self.documents:
            raise KnowledgeBaseValidationError(
                "An active knowledge base requires an attached document manifest."
            )
        if self.status in {KnowledgeBaseStatus.PREPARING, KnowledgeBaseStatus.PENDING} and (
            self.internal_index_id is not None or self.chunk_count != 0
        ):
            raise KnowledgeBaseValidationError(
                "Preparing and pending knowledge bases cannot contain index results."
            )
        if self.status is KnowledgeBaseStatus.INDEXING and self.internal_index_id is None:
            raise KnowledgeBaseValidationError(
                "An indexing knowledge base requires an internal index ID."
            )
        if self.status is KnowledgeBaseStatus.READY and self.internal_index_id is None:
            raise KnowledgeBaseValidationError("A ready knowledge base requires an internal index ID.")
        if self.status is KnowledgeBaseStatus.FAILED:
            if not isinstance(self.error_code, KnowledgeBaseErrorCode):
                raise KnowledgeBaseValidationError(
                    "A failed knowledge base requires a safe error code."
                )
        elif self.error_code is not None:
            raise KnowledgeBaseValidationError("Only failed knowledge bases may contain an error code.")
        if not is_valid_timestamp(self.created_at) or not is_valid_timestamp(self.updated_at):
            raise KnowledgeBaseValidationError(
                "Knowledge base timestamps must be finite and non-negative."
            )
        if self.updated_at < self.created_at:
            raise KnowledgeBaseValidationError("updated_at cannot precede created_at.")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise KnowledgeBaseValidationError("version must be a positive integer.")
        if self.idempotency_reservation_id is not None:
            object.__setattr__(
                self,
                "idempotency_reservation_id",
                validate_idempotency_reservation_id(self.idempotency_reservation_id),
            )


def validate_resource_id(value: object) -> str:
    if not isinstance(value, str) or _RESOURCE_ID_PATTERN.fullmatch(value) is None:
        raise KnowledgeBaseValidationError("Invalid knowledge base resource ID.")
    return value


def validate_display_name(value: object, max_length: int) -> str:
    if not isinstance(value, str):
        raise KnowledgeBaseValidationError("Display name must be text.")
    normalized = value.strip()
    if not normalized or len(normalized) > max_length:
        raise KnowledgeBaseValidationError("Display name has an invalid length.")
    if any(ord(character) < 32 for character in normalized) or any(
        character in '/\\<>:"|?*' for character in normalized
    ):
        raise KnowledgeBaseValidationError("Display name contains unsafe characters.")
    if normalized.endswith((".", " ")):
        raise KnowledgeBaseValidationError("Display name has an unsafe ending.")
    return normalized


def validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise KnowledgeBaseValidationError("Manifest path has an invalid length.")
    if "\\" in value or "\x00" in value or value.startswith("/"):
        raise KnowledgeBaseValidationError("Manifest path must be a safe POSIX relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or not path.parts:
        raise KnowledgeBaseValidationError("Manifest path must be normalized and relative.")
    for part in path.parts:
        if part in {"", ".", ".."} or part.endswith((".", " ")):
            raise KnowledgeBaseValidationError("Manifest path contains an unsafe segment.")
        if any(ord(character) < 32 for character in part) or any(
            character in '<>:"|?*' for character in part
        ):
            raise KnowledgeBaseValidationError("Manifest path contains unsafe characters.")
    return value


def validate_internal_index_id(value: object) -> str:
    if not isinstance(value, str) or _INDEX_ID_PATTERN.fullmatch(value) is None:
        raise KnowledgeBaseValidationError("Internal index ID has an invalid format.")
    return value


def validate_idempotency_reservation_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or _IDEMPOTENCY_RESERVATION_PATTERN.fullmatch(value) is None
    ):
        raise KnowledgeBaseValidationError("Idempotency reservation ID has an invalid format.")
    return value


def normalize_manifest(documents: Sequence[DocumentManifest]) -> tuple[DocumentManifest, ...]:
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise KnowledgeBaseValidationError(
            "documents must be a sequence of DocumentManifest values."
        )
    if len(documents) > MAX_DOCUMENT_MANIFEST_ITEMS:
        raise KnowledgeBaseValidationError("Document manifest exceeds the item limit.")
    normalized = tuple(documents)
    if any(not isinstance(item, DocumentManifest) for item in normalized):
        raise KnowledgeBaseValidationError("Document manifest contains an invalid item.")
    paths = [item.relative_path for item in normalized]
    if len(paths) != len(set(paths)):
        raise KnowledgeBaseValidationError("Document manifest contains duplicate relative paths.")
    return normalized


def is_valid_timestamp(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


__all__ = [
    "ALLOWED_STATUS_TRANSITIONS",
    "DocumentManifest",
    "DocumentManifestEntry",
    "KnowledgeBaseContractError",
    "KnowledgeBaseErrorCode",
    "KnowledgeBaseRecord",
    "KnowledgeBaseStatus",
    "KnowledgeBaseValidationError",
    "MAX_DOCUMENT_MANIFEST_ITEMS",
    "is_valid_timestamp",
    "normalize_manifest",
    "validate_display_name",
    "validate_idempotency_reservation_id",
    "validate_internal_index_id",
    "validate_relative_path",
    "validate_resource_id",
]
