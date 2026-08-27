"""Durable, tenant-isolated SQLite catalog for knowledge bases."""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, cast

from .tenancy import Principal, TenantId
from rag_system.knowledge_base_contracts import (
    ALLOWED_STATUS_TRANSITIONS,
    DocumentManifest,
    DocumentManifestEntry,
    KnowledgeBaseContractError as CatalogError,
    KnowledgeBaseErrorCode,
    KnowledgeBaseRecord,
    KnowledgeBaseStatus,
    KnowledgeBaseValidationError as CatalogValidationError,
    MAX_DOCUMENT_MANIFEST_ITEMS,
    is_valid_timestamp,
    normalize_manifest,
    validate_display_name,
    validate_idempotency_reservation_id,
    validate_internal_index_id,
    validate_resource_id,
)


_SCHEMA_VERSION = 4
_MAX_LIST_LIMIT = 100
_MAX_LIST_OFFSET = 10_000
_MAX_MANIFEST_JSON_BYTES = 8 * 1024 * 1024


class CatalogSchemaError(CatalogError):
    """The on-disk schema or stored data violates the catalog contract."""

    def __init__(self) -> None:
        super().__init__("Catalog schema or stored data is invalid.")


class CatalogStorageError(CatalogError):
    """A sanitized SQLite operation failure."""

    def __init__(self) -> None:
        super().__init__("Catalog storage operation failed.")


class KnowledgeBaseUnavailableError(CatalogError):
    """The same denial for missing and cross-tenant resources."""

    def __init__(self) -> None:
        super().__init__("Knowledge base is unavailable.")


class InvalidStatusTransitionError(CatalogError):
    def __init__(self, current: KnowledgeBaseStatus, target: KnowledgeBaseStatus) -> None:
        super().__init__(f"Invalid knowledge base status transition: {current.value} -> {target.value}.")


class KnowledgeBaseCatalog:
    """A thread-safe catalog using short SQLite transactions per operation.

    WAL provides process-safe concurrent readers while ``BEGIN IMMEDIATE``
    serializes state transitions.  The object keeps no database connection or
    plaintext document content in memory between calls.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        clock: Callable[[], float] = time.time,
        timeout_seconds: float = 5.0,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 60:
            raise CatalogValidationError("timeout_seconds must be between 0 and 60.")
        path = Path(database_path)
        if path.exists() and not path.is_file():
            raise CatalogValidationError("database_path must reference a file.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._database_path = path.resolve()
        self._clock = clock
        self._timeout_seconds = float(timeout_seconds)
        self._write_lock = RLock()
        self._initialize()

    @property
    def database_path(self) -> Path:
        return self._database_path

    def create(
        self,
        principal: Principal,
        display_name: str,
        *,
        idempotency_reservation_id: str | None = None,
    ) -> KnowledgeBaseRecord:
        tenant_id = _principal_tenant(principal)
        clean_name = validate_display_name(display_name, 200)
        manifest_json = _encode_manifest(())
        created_at = self._now()
        reservation_id = (
            validate_idempotency_reservation_id(idempotency_reservation_id)
            if idempotency_reservation_id is not None
            else None
        )

        with self._write_lock, self._write_transaction() as connection:
            for _ in range(8):
                resource_id = _new_resource_id()
                try:
                    connection.execute(
                        """
                        INSERT INTO knowledge_bases (
                            resource_id, tenant_id, display_name, status,
                            internal_index_id, manifest_json, document_count,
                            total_bytes, chunk_count, error_code, created_at,
                            updated_at, version, idempotency_reservation_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            resource_id,
                            tenant_id.value,
                            clean_name,
                            KnowledgeBaseStatus.PREPARING.value,
                            None,
                            manifest_json,
                            0,
                            0,
                            0,
                            None,
                            created_at,
                            created_at,
                            1,
                            reservation_id,
                        ),
                    )
                except sqlite3.IntegrityError:
                    continue
                row = self._select_row(connection, tenant_id, resource_id)
                if row is None:
                    raise CatalogStorageError()
                return _record_from_row(row)
        raise CatalogStorageError()

    def get(self, principal: Principal, resource_id: str) -> KnowledgeBaseRecord:
        tenant_id = _principal_tenant(principal)
        clean_id = _safe_resource_lookup_id(resource_id)
        with self._read_connection() as connection:
            row = self._select_row(connection, tenant_id, clean_id)
        if row is None:
            raise KnowledgeBaseUnavailableError()
        return _record_from_row(row)

    def find_by_idempotency_reservation(
        self,
        principal: Principal,
        reservation_id: str,
    ) -> KnowledgeBaseRecord | None:
        """Find the tenant-owned resource for crash recovery, if it exists."""

        tenant_id = _principal_tenant(principal)
        clean_reservation_id = validate_idempotency_reservation_id(reservation_id)
        with self._read_connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM knowledge_bases
                WHERE tenant_id = ? AND idempotency_reservation_id = ?
                """,
                (tenant_id.value, clean_reservation_id),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def list(
        self,
        principal: Principal,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[KnowledgeBaseRecord, ...]:
        tenant_id = _principal_tenant(principal)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LIST_LIMIT:
            raise CatalogValidationError(f"limit must be between 1 and {_MAX_LIST_LIMIT}.")
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= _MAX_LIST_OFFSET:
            raise CatalogValidationError(f"offset must be between 0 and {_MAX_LIST_OFFSET}.")
        with self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM knowledge_bases
                WHERE tenant_id = ?
                ORDER BY updated_at DESC, resource_id DESC
                LIMIT ? OFFSET ?
                """,
                (tenant_id.value, limit, offset),
            ).fetchall()
        return tuple(_record_from_row(row) for row in rows)

    def list_after(
        self,
        principal: Principal,
        *,
        updated_at: float,
        resource_id: str,
        limit: int = 50,
    ) -> tuple[KnowledgeBaseRecord, ...]:
        """Return the next stable page without a costly deep SQL offset."""

        tenant_id = _principal_tenant(principal)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LIST_LIMIT:
            raise CatalogValidationError(f"limit must be between 1 and {_MAX_LIST_LIMIT}.")
        if not is_valid_timestamp(updated_at):
            raise CatalogValidationError("cursor timestamp is invalid.")
        clean_resource_id = _safe_resource_lookup_id(resource_id)
        with self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM knowledge_bases
                WHERE tenant_id = ?
                  AND (updated_at < ? OR (updated_at = ? AND resource_id < ?))
                ORDER BY updated_at DESC, resource_id DESC
                LIMIT ?
                """,
                (tenant_id.value, updated_at, updated_at, clean_resource_id, limit),
            ).fetchall()
        return tuple(_record_from_row(row) for row in rows)

    def attach_manifest(
        self,
        principal: Principal,
        resource_id: str,
        documents: Sequence[DocumentManifest],
    ) -> KnowledgeBaseRecord:
        tenant_id = _principal_tenant(principal)
        clean_id = _safe_resource_lookup_id(resource_id)
        manifest = normalize_manifest(documents)
        if not manifest:
            raise CatalogValidationError("Document manifest cannot be empty.")
        manifest_json = _encode_manifest(manifest)
        with self._write_lock, self._write_transaction() as connection:
            current = self._owned_record(connection, tenant_id, clean_id)
            if current.status is not KnowledgeBaseStatus.PREPARING:
                raise InvalidStatusTransitionError(
                    current.status,
                    KnowledgeBaseStatus.PREPARING,
                )
            if current.documents:
                if current.documents == manifest:
                    return current
                raise CatalogValidationError(
                    "Document manifest is immutable once attached."
                )
            connection.execute(
                """
                UPDATE knowledge_bases
                SET manifest_json = ?, document_count = ?, total_bytes = ?,
                    updated_at = ?, version = version + 1
                WHERE resource_id = ? AND tenant_id = ?
                """,
                (
                    manifest_json,
                    len(manifest),
                    sum(item.size_bytes for item in manifest),
                    self._now(),
                    clean_id,
                    tenant_id.value,
                ),
            )
            return self._owned_record(connection, tenant_id, clean_id)

    def transition(
        self,
        principal: Principal,
        resource_id: str,
        target: KnowledgeBaseStatus,
        *,
        internal_index_id: str | None = None,
        chunk_count: int | None = None,
        error_code: KnowledgeBaseErrorCode | None = None,
    ) -> KnowledgeBaseRecord:
        tenant_id = _principal_tenant(principal)
        clean_id = _safe_resource_lookup_id(resource_id)
        if not isinstance(target, KnowledgeBaseStatus):
            raise CatalogValidationError("target must be a KnowledgeBaseStatus.")

        with self._write_lock, self._write_transaction() as connection:
            current = self._owned_record(connection, tenant_id, clean_id)
            if target not in ALLOWED_STATUS_TRANSITIONS[current.status]:
                raise InvalidStatusTransitionError(current.status, target)

            new_index_id = current.internal_index_id
            new_chunk_count = current.chunk_count
            new_error_code: KnowledgeBaseErrorCode | None = None

            if target is KnowledgeBaseStatus.PENDING:
                if internal_index_id is not None or chunk_count is not None or error_code is not None:
                    raise CatalogValidationError(
                        "Pending rejects index, chunk, and error overrides."
                    )
                if not current.documents:
                    raise CatalogValidationError(
                        "Pending requires an attached document manifest."
                    )
                new_index_id = None
                new_chunk_count = 0
            elif target is KnowledgeBaseStatus.INDEXING:
                if internal_index_id is None or chunk_count is not None or error_code is not None:
                    raise CatalogValidationError(
                        "Indexing requires internal_index_id and rejects chunk_count/error_code."
                    )
                new_index_id = validate_internal_index_id(internal_index_id)
                new_chunk_count = 0
            elif target is KnowledgeBaseStatus.READY:
                if internal_index_id is not None or error_code is not None:
                    raise CatalogValidationError(
                        "Ready uses the existing internal index and rejects error_code."
                    )
                if current.internal_index_id is None:
                    raise CatalogValidationError("Ready requires an existing internal index ID.")
                if isinstance(chunk_count, bool) or not isinstance(chunk_count, int) or chunk_count < 0:
                    raise CatalogValidationError("Ready requires a non-negative chunk_count.")
                new_chunk_count = chunk_count
            elif target is KnowledgeBaseStatus.FAILED:
                if internal_index_id is not None or chunk_count is not None:
                    raise CatalogValidationError("Failed rejects index and chunk overrides.")
                if not isinstance(error_code, KnowledgeBaseErrorCode):
                    raise CatalogValidationError("Failed requires a safe KnowledgeBaseErrorCode.")
                new_error_code = error_code
            elif target is KnowledgeBaseStatus.CANCELLING:
                if internal_index_id is not None or chunk_count is not None or error_code is not None:
                    raise CatalogValidationError(
                        "Cancelling rejects index, chunk, and error overrides."
                    )
            elif target is KnowledgeBaseStatus.DELETING:
                if internal_index_id is not None or chunk_count is not None or error_code is not None:
                    raise CatalogValidationError("Deleting rejects index, chunk, and error overrides.")

            connection.execute(
                """
                UPDATE knowledge_bases
                SET status = ?, internal_index_id = ?, chunk_count = ?,
                    error_code = ?, updated_at = ?, version = version + 1
                WHERE resource_id = ? AND tenant_id = ?
                """,
                (
                    target.value,
                    new_index_id,
                    new_chunk_count,
                    new_error_code.value if new_error_code else None,
                    self._now(),
                    clean_id,
                    tenant_id.value,
                ),
            )
            return self._owned_record(connection, tenant_id, clean_id)

    def delete(
        self,
        principal: Principal,
        resource_id: str,
    ) -> tuple[DocumentManifest, ...]:
        """Delete a DELETING record and return its immutable cleanup manifest."""

        tenant_id = _principal_tenant(principal)
        clean_id = _safe_resource_lookup_id(resource_id)
        with self._write_lock, self._write_transaction() as connection:
            current = self._owned_record(connection, tenant_id, clean_id)
            if current.status is not KnowledgeBaseStatus.DELETING:
                raise InvalidStatusTransitionError(current.status, KnowledgeBaseStatus.DELETING)
            cursor = connection.execute(
                "DELETE FROM knowledge_bases WHERE resource_id = ? AND tenant_id = ?",
                (clean_id, tenant_id.value),
            )
            if cursor.rowcount != 1:
                raise KnowledgeBaseUnavailableError()
            return current.documents

    def _initialize(self) -> None:
        with self._write_lock, self._write_transaction(validate_schema=False) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 0:
                existing = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'knowledge_bases'"
                ).fetchone()
                if existing is not None:
                    raise CatalogSchemaError()
                connection.execute(_CREATE_TABLE_SQL)
                self._create_indexes(connection)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif version in {2, 3}:
                self._migrate_to_v4(connection, source_version=version)
            elif version != _SCHEMA_VERSION:
                raise CatalogSchemaError()
            self._validate_schema(connection)

    def _migrate_to_v4(
        self,
        connection: sqlite3.Connection,
        *,
        source_version: int,
    ) -> None:
        """Rebuild a supported legacy table with the explicit preparing state."""

        self._validate_schema(connection)
        names = tuple(name for name, *_ in _EXPECTED_COLUMNS)
        column_names = ", ".join(names)
        selected_columns = ", ".join(
            (
                "CASE WHEN status = 'pending' AND manifest_json = '[]' "
                "THEN 'preparing' ELSE status END AS status"
                if name == "status"
                else name
            )
            for name in names
        )
        legacy_table = f"knowledge_bases_v{source_version}"
        connection.execute(
            f"ALTER TABLE knowledge_bases RENAME TO {legacy_table}"
        )
        connection.execute(_CREATE_TABLE_SQL)
        connection.execute(
            f"INSERT INTO knowledge_bases ({column_names}) "
            f"SELECT {selected_columns} FROM {legacy_table}"
        )
        connection.execute(f"DROP TABLE {legacy_table}")
        self._create_indexes(connection)
        connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    @staticmethod
    def _create_indexes(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE INDEX idx_knowledge_bases_tenant_updated "
            "ON knowledge_bases (tenant_id, updated_at DESC, resource_id DESC)"
        )
        connection.execute(
            "CREATE UNIQUE INDEX idx_knowledge_bases_idempotency "
            "ON knowledge_bases (tenant_id, idempotency_reservation_id) "
            "WHERE idempotency_reservation_id IS NOT NULL"
        )

    def _validate_schema(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute("PRAGMA table_info(knowledge_bases)").fetchall()
        actual = tuple(
            (row["name"], row["type"].upper(), int(row["notnull"]), int(row["pk"]))
            for row in rows
        )
        indexes = {
            str(row["name"]): (int(row["unique"]), int(row["partial"]))
            for row in connection.execute("PRAGMA index_list(knowledge_bases)").fetchall()
        }
        if actual != _EXPECTED_COLUMNS or any(
            indexes.get(name) != (unique, partial)
            for name, unique, partial in _EXPECTED_CUSTOM_INDEXES
        ):
            raise CatalogSchemaError()
        for name, _, _ in _EXPECTED_CUSTOM_INDEXES:
            columns = tuple(
                str(row["name"])
                for row in connection.execute(f"PRAGMA index_info({name})").fetchall()
            )
            if columns != _EXPECTED_INDEX_COLUMNS[name]:
                raise CatalogSchemaError()

    def _owned_record(
        self,
        connection: sqlite3.Connection,
        tenant_id: TenantId,
        resource_id: str,
    ) -> KnowledgeBaseRecord:
        row = self._select_row(connection, tenant_id, resource_id)
        if row is None:
            raise KnowledgeBaseUnavailableError()
        return _record_from_row(row)

    @staticmethod
    def _select_row(
        connection: sqlite3.Connection,
        tenant_id: TenantId,
        resource_id: str,
    ) -> sqlite3.Row | None:
        return cast(
            sqlite3.Row | None,
            connection.execute(
                "SELECT * FROM knowledge_bases WHERE resource_id = ? AND tenant_id = ?",
                (resource_id, tenant_id.value),
            ).fetchone(),
        )

    def _now(self) -> float:
        value = float(self._clock())
        if not is_valid_timestamp(value):
            raise CatalogValidationError("clock returned an invalid timestamp.")
        return value

    def _connect(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self._database_path,
                timeout=self._timeout_seconds,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {int(self._timeout_seconds * 1000)}")
            mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
            if mode != "wal":
                raise CatalogStorageError()
            return connection
        except CatalogStorageError:
            if connection is not None:
                connection.close()
            raise
        except sqlite3.Error as exc:
            if connection is not None:
                connection.close()
            raise CatalogStorageError() from exc

    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        except sqlite3.Error as exc:
            raise CatalogStorageError() from exc
        finally:
            connection.close()

    @contextmanager
    def _write_transaction(
        self,
        *,
        validate_schema: bool = True,
    ) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if validate_schema:
                self._validate_schema(connection)
            yield connection
            connection.commit()
        except CatalogError:
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise CatalogStorageError() from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


_CREATE_TABLE_SQL = """
CREATE TABLE knowledge_bases (
    resource_id TEXT PRIMARY KEY NOT NULL,
    tenant_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN (
            'preparing', 'pending', 'indexing', 'cancelling',
            'ready', 'failed', 'deleting'
        )
    ),
    internal_index_id TEXT,
    manifest_json TEXT NOT NULL,
    document_count INTEGER NOT NULL CHECK (document_count >= 0),
    total_bytes INTEGER NOT NULL CHECK (total_bytes >= 0),
    chunk_count INTEGER NOT NULL CHECK (chunk_count >= 0),
    error_code TEXT CHECK (
        error_code IS NULL OR error_code IN (
            'content_rejected', 'ingestion_failed', 'index_build_failed',
            'index_storage_failed', 'upstream_unavailable', 'internal_error',
            'index_cancelled'
        )
    ),
    created_at REAL NOT NULL CHECK (created_at >= 0),
    updated_at REAL NOT NULL CHECK (updated_at >= created_at),
    version INTEGER NOT NULL CHECK (version >= 1),
    idempotency_reservation_id TEXT
)
"""


_EXPECTED_COLUMNS = (
    ("resource_id", "TEXT", 1, 1),
    ("tenant_id", "TEXT", 1, 0),
    ("display_name", "TEXT", 1, 0),
    ("status", "TEXT", 1, 0),
    ("internal_index_id", "TEXT", 0, 0),
    ("manifest_json", "TEXT", 1, 0),
    ("document_count", "INTEGER", 1, 0),
    ("total_bytes", "INTEGER", 1, 0),
    ("chunk_count", "INTEGER", 1, 0),
    ("error_code", "TEXT", 0, 0),
    ("created_at", "REAL", 1, 0),
    ("updated_at", "REAL", 1, 0),
    ("version", "INTEGER", 1, 0),
    ("idempotency_reservation_id", "TEXT", 0, 0),
)

_EXPECTED_CUSTOM_INDEXES = (
    ("idx_knowledge_bases_tenant_updated", 0, 0),
    ("idx_knowledge_bases_idempotency", 1, 1),
)

_EXPECTED_INDEX_COLUMNS = {
    "idx_knowledge_bases_tenant_updated": ("tenant_id", "updated_at", "resource_id"),
    "idx_knowledge_bases_idempotency": ("tenant_id", "idempotency_reservation_id"),
}


def _principal_tenant(principal: Principal) -> TenantId:
    if not isinstance(principal, Principal):
        raise TypeError("principal must be a Principal")
    return principal.tenant_id


def _new_resource_id() -> str:
    value = f"kb_{secrets.token_hex(16)}"
    validate_resource_id(value)
    return value


def _safe_resource_lookup_id(value: object) -> str:
    try:
        return validate_resource_id(value)
    except CatalogValidationError:
        raise KnowledgeBaseUnavailableError() from None


def _encode_manifest(documents: tuple[DocumentManifest, ...]) -> str:
    payload = [
        {
            "display_name": item.display_name,
            "relative_path": item.relative_path,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for item in documents
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if len(encoded.encode("utf-8")) > _MAX_MANIFEST_JSON_BYTES:
        raise CatalogValidationError("Encoded document manifest exceeds the size limit.")
    return encoded


def _decode_manifest(value: object) -> tuple[DocumentManifest, ...]:
    if not isinstance(value, str) or len(value.encode("utf-8")) > _MAX_MANIFEST_JSON_BYTES:
        raise CatalogSchemaError()
    try:
        payload = json.loads(
            value,
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda _constant: (_ for _ in ()).throw(ValueError()),
        )
        if not isinstance(payload, list) or len(payload) > MAX_DOCUMENT_MANIFEST_ITEMS:
            raise ValueError
        documents: list[DocumentManifest] = []
        required_keys = {"display_name", "relative_path", "size_bytes", "sha256"}
        for item in payload:
            if not isinstance(item, dict) or set(item) != required_keys:
                raise ValueError
            documents.append(
                DocumentManifest(
                    display_name=item["display_name"],
                    relative_path=item["relative_path"],
                    size_bytes=item["size_bytes"],
                    sha256=item["sha256"],
                )
            )
        return normalize_manifest(documents)
    except (CatalogError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CatalogSchemaError() from exc


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _record_from_row(row: sqlite3.Row) -> KnowledgeBaseRecord:
    try:
        documents = _decode_manifest(row["manifest_json"])
        error_value = row["error_code"]
        return KnowledgeBaseRecord(
            resource_id=row["resource_id"],
            tenant_id=TenantId(row["tenant_id"]),
            display_name=row["display_name"],
            status=KnowledgeBaseStatus(row["status"]),
            internal_index_id=row["internal_index_id"],
            documents=documents,
            document_count=row["document_count"],
            total_bytes=row["total_bytes"],
            chunk_count=row["chunk_count"],
            error_code=KnowledgeBaseErrorCode(error_value) if error_value is not None else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            version=row["version"],
            idempotency_reservation_id=row["idempotency_reservation_id"],
        )
    except CatalogSchemaError:
        raise
    except (CatalogError, KeyError, TypeError, ValueError) as exc:
        raise CatalogSchemaError() from exc


__all__ = [
    "CatalogError",
    "CatalogSchemaError",
    "CatalogStorageError",
    "CatalogValidationError",
    "DocumentManifest",
    "DocumentManifestEntry",
    "InvalidStatusTransitionError",
    "KnowledgeBaseCatalog",
    "KnowledgeBaseErrorCode",
    "KnowledgeBaseRecord",
    "KnowledgeBaseStatus",
    "KnowledgeBaseUnavailableError",
]
