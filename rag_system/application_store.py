"""Durable SQLite repository for versioned application-kernel records."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from threading import RLock
from typing import Any, cast

from rag_system.application_contracts import (
    AnswerPolicy,
    Application,
    ApplicationAuditEventType,
    ApplicationContractError,
    ApplicationDraft,
    ApplicationKind,
    ApplicationRevision,
    ApplicationStatus,
    ApplicationValidationError,
    AuditEvent,
    Deployment,
    DeploymentEnvironment,
    KnowledgeChatConfiguration,
    Project,
    ResourceAccessMode,
    ResourceBinding,
    ResourceKind,
    RetrievalProfile,
    SessionPolicy,
    DEFAULT_MODEL_PROFILE_ID,
    is_valid_timestamp,
    validate_application_id,
    validate_deployment_id,
    validate_project_id,
    validate_revision_id,
)
from rag_system.application_evaluation import (
    MAX_EVALUATION_REPORT_BYTES,
    ApplicationEvaluationError,
    ApplicationEvaluationReport,
)
from rag_system.sqlite_support import SqliteDatabase
from rag_system.tenancy import Principal, TenantId


_SCHEMA_VERSION = 7
_MAX_LIST_LIMIT = 100
_MAX_CONFIGURATION_BYTES = 64 * 1024
_UNSET = object()


class ApplicationStoreError(ApplicationContractError):
    """Base error for durable application-store failures."""


class ApplicationStoreSchemaError(ApplicationStoreError):
    def __init__(self) -> None:
        super().__init__("Application store schema or stored data is invalid.")


class ApplicationStoreStorageError(ApplicationStoreError):
    def __init__(self) -> None:
        super().__init__("Application store operation failed.")


class ProjectUnavailableError(ApplicationStoreError):
    def __init__(self) -> None:
        super().__init__("Project is unavailable.")


class ApplicationUnavailableError(ApplicationStoreError):
    def __init__(self) -> None:
        super().__init__("Application is unavailable.")


class ApplicationRevisionUnavailableError(ApplicationStoreError):
    def __init__(self) -> None:
        super().__init__("Application revision is unavailable.")


class ApplicationDraftConflictError(ApplicationStoreError):
    def __init__(self) -> None:
        super().__init__("Application draft has changed.")


class ApplicationPublishConflictError(ApplicationStoreError):
    def __init__(self) -> None:
        super().__init__("Application publication has changed.")


class DeploymentUnavailableError(ApplicationStoreError):
    def __init__(self) -> None:
        super().__init__("Deployment is unavailable.")


class ApplicationStore:
    """Short-transaction, tenant-scoped persistence for the application kernel."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ApplicationValidationError("timeout_seconds must be between 0 and 60.")
        if not 0 < timeout_seconds <= 60:
            raise ApplicationValidationError("timeout_seconds must be between 0 and 60.")
        path = Path(database_path)
        if path.exists() and not path.is_file():
            raise ApplicationValidationError("database_path must reference a file.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._database_path = path.resolve()
        self._database = SqliteDatabase(self._database_path, timeout_seconds=float(timeout_seconds))
        self._write_lock = RLock()
        self._initialize()

    @property
    def database_path(self) -> Path:
        return self._database_path

    def create_project(self, principal: Principal, project: Project) -> Project:
        tenant_id = _principal_tenant(principal)
        _require_tenant_record(tenant_id, project.tenant_id)
        with self._write_lock, self._write_transaction() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO projects (project_id, tenant_id, display_name, description, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project.project_id,
                        tenant_id.value,
                        project.display_name,
                        project.description,
                        project.created_at,
                        project.updated_at,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ApplicationStoreStorageError() from error
        return project

    def get_project(self, principal: Principal, project_id: str) -> Project:
        tenant_id = _principal_tenant(principal)
        clean_id = _safe_project_lookup_id(project_id)
        with self._read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE project_id = ? AND tenant_id = ?",
                (clean_id, tenant_id.value),
            ).fetchone()
        if row is None:
            raise ProjectUnavailableError()
        return _project_from_row(row)

    def list_projects(self, principal: Principal, *, limit: int = 50) -> tuple[Project, ...]:
        tenant_id = _principal_tenant(principal)
        clean_limit = _validate_limit(limit)
        with self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM projects WHERE tenant_id = ?
                ORDER BY updated_at DESC, project_id DESC LIMIT ?
                """,
                (tenant_id.value, clean_limit),
            ).fetchall()
        return tuple(_project_from_row(row) for row in rows)

    def create_application(self, principal: Principal, application: Application) -> Application:
        tenant_id = _principal_tenant(principal)
        _require_tenant_record(tenant_id, application.tenant_id)
        with self._write_lock, self._write_transaction() as connection:
            self._require_project(connection, tenant_id, application.project_id)
            try:
                connection.execute(
                    """
                    INSERT INTO applications (
                        application_id, tenant_id, project_id, display_name, application_kind,
                        active_revision_id, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        application.application_id,
                        tenant_id.value,
                        application.project_id,
                        application.display_name,
                        application.application_kind.value,
                        application.active_revision_id,
                        application.status.value,
                        application.created_at,
                        application.updated_at,
                    ),
                )
                connection.execute(
                    """INSERT INTO application_drafts (
                    application_id, version, configuration_json, updated_at, updated_by, change_summary
                    ) VALUES (?, 0, NULL, ?, ?, '')""",
                    (application.application_id, application.updated_at, principal.subject),
                )
            except sqlite3.IntegrityError as error:
                raise ApplicationStoreStorageError() from error
        return application

    def get_application(self, principal: Principal, application_id: str) -> Application:
        tenant_id = _principal_tenant(principal)
        clean_id = _safe_application_lookup_id(application_id)
        with self._read_connection() as connection:
            row = self._select_application(connection, tenant_id, clean_id)
        if row is None:
            raise ApplicationUnavailableError()
        return _application_from_row(row)

    def list_applications(
        self,
        principal: Principal,
        project_id: str,
        *,
        limit: int = 50,
    ) -> tuple[Application, ...]:
        tenant_id = _principal_tenant(principal)
        clean_project_id = _safe_project_lookup_id(project_id)
        clean_limit = _validate_limit(limit)
        with self._read_connection() as connection:
            self._require_project(connection, tenant_id, clean_project_id)
            rows = connection.execute(
                """
                SELECT * FROM applications WHERE tenant_id = ? AND project_id = ?
                ORDER BY updated_at DESC, application_id DESC LIMIT ?
                """,
                (tenant_id.value, clean_project_id, clean_limit),
            ).fetchall()
        return tuple(_application_from_row(row) for row in rows)

    def archive_application(
        self,
        principal: Principal,
        application_id: str,
        audit_event: AuditEvent,
        *,
        updated_at: float,
    ) -> Application:
        tenant_id = _principal_tenant(principal)
        clean_id = _safe_application_lookup_id(application_id)
        if not is_valid_timestamp(updated_at):
            raise ApplicationValidationError("updated_at must be finite and non-negative.")
        if audit_event.event_type is not ApplicationAuditEventType.APPLICATION_ARCHIVED:
            raise ApplicationValidationError("Archiving requires an application-archived audit event.")
        with self._write_lock, self._write_transaction() as connection:
            application = self._require_application(connection, tenant_id, clean_id)
            if updated_at < application.updated_at:
                raise ApplicationValidationError("updated_at cannot move backwards.")
            if audit_event.tenant_id != tenant_id or audit_event.application_id != clean_id:
                raise ApplicationUnavailableError()
            connection.execute(
                "UPDATE applications SET status = ?, updated_at = ? WHERE application_id = ?",
                (ApplicationStatus.ARCHIVED.value, updated_at, clean_id),
            )
            self._insert_audit_event(connection, audit_event)
            row = self._select_application(connection, tenant_id, application.application_id)
        if row is None:
            raise ApplicationUnavailableError()
        return _application_from_row(row)

    def get_draft(self, principal: Principal, application_id: str) -> ApplicationDraft:
        tenant_id = _principal_tenant(principal)
        clean_id = _safe_application_lookup_id(application_id)
        with self._read_connection() as connection:
            self._require_application(connection, tenant_id, clean_id)
            row = connection.execute(
                "SELECT * FROM application_drafts WHERE application_id = ?", (clean_id,)
            ).fetchone()
        if row is None:
            raise ApplicationStoreSchemaError()
        return _draft_from_row(row)

    def update_draft(
        self,
        principal: Principal,
        draft: ApplicationDraft,
        audit_event: AuditEvent,
        *,
        expected_version: int,
    ) -> ApplicationDraft:
        tenant_id = _principal_tenant(principal)
        if isinstance(expected_version, bool) or not isinstance(expected_version, int):
            raise ApplicationValidationError("expected draft version must be an integer.")
        if audit_event.event_type is not ApplicationAuditEventType.DRAFT_UPDATED:
            raise ApplicationValidationError("Draft updates require a draft-updated audit event.")
        if draft.configuration is None:
            raise ApplicationValidationError("Draft updates require a configuration.")
        encoded = _encode_configuration(draft.configuration)
        with self._write_lock, self._write_transaction() as connection:
            application = self._require_application(connection, tenant_id, draft.application_id)
            if application.status is not ApplicationStatus.ACTIVE:
                raise ApplicationValidationError("Archived applications cannot update drafts.")
            result = connection.execute(
                """UPDATE application_drafts
                SET version = ?, configuration_json = ?, updated_at = ?, updated_by = ?, change_summary = ?
                WHERE application_id = ? AND version = ?""",
                (
                    draft.version,
                    encoded,
                    draft.updated_at,
                    draft.updated_by,
                    draft.change_summary,
                    draft.application_id,
                    expected_version,
                ),
            )
            if result.rowcount != 1:
                raise ApplicationDraftConflictError()
            if audit_event.tenant_id != tenant_id or audit_event.application_id != draft.application_id:
                raise ApplicationUnavailableError()
            self._insert_audit_event(connection, audit_event)
            row = connection.execute(
                "SELECT * FROM application_drafts WHERE application_id = ?", (draft.application_id,)
            ).fetchone()
        if row is None:
            raise ApplicationStoreSchemaError()
        return _draft_from_row(row)

    def create_revision(
        self,
        principal: Principal,
        revision: ApplicationRevision,
        bindings: Sequence[ResourceBinding],
    ) -> ApplicationRevision:
        tenant_id = _principal_tenant(principal)
        clean_bindings = _normalize_bindings(revision, bindings)
        configuration_json = _encode_configuration(revision.configuration)
        with self._write_lock, self._write_transaction() as connection:
            application = self._require_application(connection, tenant_id, revision.application_id)
            _validate_revision_for_application(application, revision, clean_bindings)
            try:
                connection.execute(
                    """
                    INSERT INTO application_revisions (
                        revision_id, application_id, revision_number, configuration_schema_version,
                        configuration_json, created_at, created_by, change_summary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        revision.revision_id,
                        revision.application_id,
                        revision.revision_number,
                        revision.configuration_schema_version,
                        configuration_json,
                        revision.created_at,
                        revision.created_by,
                        revision.change_summary,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO resource_bindings (
                        binding_id, application_id, revision_id, resource_kind,
                        resource_id, access_mode, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    tuple(
                        (
                            binding.binding_id,
                            binding.application_id,
                            binding.revision_id,
                            binding.resource_kind.value,
                            binding.resource_id,
                            binding.access_mode.value,
                            binding.created_at,
                        )
                        for binding in clean_bindings
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ApplicationStoreStorageError() from error
        return revision

    def get_revision(
        self,
        principal: Principal,
        application_id: str,
        revision_id: str,
    ) -> ApplicationRevision:
        tenant_id = _principal_tenant(principal)
        clean_application_id = _safe_application_lookup_id(application_id)
        clean_revision_id = _safe_revision_lookup_id(revision_id)
        with self._read_connection() as connection:
            row = self._select_revision(connection, tenant_id, clean_application_id, clean_revision_id)
        if row is None:
            raise ApplicationRevisionUnavailableError()
        return _revision_from_row(row)

    def list_revisions(
        self,
        principal: Principal,
        application_id: str,
        *,
        limit: int = 50,
    ) -> tuple[ApplicationRevision, ...]:
        tenant_id = _principal_tenant(principal)
        clean_application_id = _safe_application_lookup_id(application_id)
        clean_limit = _validate_limit(limit)
        with self._read_connection() as connection:
            self._require_application(connection, tenant_id, clean_application_id)
            rows = connection.execute(
                """
                SELECT r.* FROM application_revisions AS r
                WHERE r.application_id = ? ORDER BY r.revision_number DESC LIMIT ?
                """,
                (clean_application_id, clean_limit),
            ).fetchall()
        return tuple(_revision_from_row(row) for row in rows)

    def list_bindings(
        self,
        principal: Principal,
        application_id: str,
        revision_id: str,
    ) -> tuple[ResourceBinding, ...]:
        tenant_id = _principal_tenant(principal)
        clean_application_id = _safe_application_lookup_id(application_id)
        clean_revision_id = _safe_revision_lookup_id(revision_id)
        with self._read_connection() as connection:
            if self._select_revision(connection, tenant_id, clean_application_id, clean_revision_id) is None:
                raise ApplicationRevisionUnavailableError()
            rows = connection.execute(
                """
                SELECT * FROM resource_bindings WHERE revision_id = ?
                ORDER BY resource_kind, resource_id, binding_id
                """,
                (clean_revision_id,),
            ).fetchall()
        return tuple(_binding_from_row(row) for row in rows)

    def create_deployment(self, principal: Principal, deployment: Deployment) -> Deployment:
        tenant_id = _principal_tenant(principal)
        with self._write_lock, self._write_transaction() as connection:
            if self._select_revision(
                connection,
                tenant_id,
                deployment.application_id,
                deployment.revision_id,
            ) is None:
                raise ApplicationRevisionUnavailableError()
            try:
                connection.execute(
                    """
                    INSERT INTO deployments (
                        deployment_id, application_id, revision_id, environment, deployed_at, deployed_by
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        deployment.deployment_id,
                        deployment.application_id,
                        deployment.revision_id,
                        deployment.environment.value,
                        deployment.deployed_at,
                        deployment.deployed_by,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ApplicationStoreStorageError() from error
        return deployment

    def publish(
        self,
        principal: Principal,
        deployment: Deployment,
        audit_event: AuditEvent,
        *,
        updated_at: float,
        expected_active_revision_id: str | None | object = _UNSET,
    ) -> Application:
        """Atomically activate a revision, retain its deployment, and audit it."""

        tenant_id = _principal_tenant(principal)
        if not is_valid_timestamp(updated_at):
            raise ApplicationValidationError("updated_at must be finite and non-negative.")
        with self._write_lock, self._write_transaction() as connection:
            application = self._require_application(connection, tenant_id, deployment.application_id)
            if application.status is not ApplicationStatus.ACTIVE:
                raise ApplicationValidationError("Archived applications cannot be published.")
            if (
                expected_active_revision_id is not _UNSET
                and application.active_revision_id != expected_active_revision_id
            ):
                raise ApplicationPublishConflictError()
            if self._select_revision(
                connection, tenant_id, deployment.application_id, deployment.revision_id
            ) is None:
                raise ApplicationRevisionUnavailableError()
            _validate_publish_audit_event(tenant_id, deployment, audit_event)
            try:
                connection.execute(
                    """
                    UPDATE applications SET active_revision_id = ?, updated_at = ?
                    WHERE application_id = ? AND tenant_id = ?
                    """,
                    (deployment.revision_id, updated_at, deployment.application_id, tenant_id.value),
                )
                connection.execute(
                    """
                    INSERT INTO deployments (
                        deployment_id, application_id, revision_id, environment, deployed_at, deployed_by
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        deployment.deployment_id,
                        deployment.application_id,
                        deployment.revision_id,
                        deployment.environment.value,
                        deployment.deployed_at,
                        deployment.deployed_by,
                    ),
                )
                self._insert_audit_event(connection, audit_event)
            except sqlite3.IntegrityError as error:
                raise ApplicationStoreStorageError() from error
            return self._require_application(connection, tenant_id, deployment.application_id)

    def get_deployment(self, principal: Principal, deployment_id: str) -> Deployment:
        tenant_id = _principal_tenant(principal)
        clean_id = _safe_deployment_lookup_id(deployment_id)
        with self._read_connection() as connection:
            row = connection.execute(
                """
                SELECT d.* FROM deployments AS d
                JOIN applications AS a ON a.application_id = d.application_id
                WHERE d.deployment_id = ? AND a.tenant_id = ?
                """,
                (clean_id, tenant_id.value),
            ).fetchone()
        if row is None:
            raise DeploymentUnavailableError()
        return _deployment_from_row(row)

    def list_deployments(
        self,
        principal: Principal,
        application_id: str,
        *,
        limit: int = 50,
    ) -> tuple[Deployment, ...]:
        tenant_id = _principal_tenant(principal)
        clean_application_id = _safe_application_lookup_id(application_id)
        clean_limit = _validate_limit(limit)
        with self._read_connection() as connection:
            self._require_application(connection, tenant_id, clean_application_id)
            rows = connection.execute(
                """
                SELECT * FROM deployments WHERE application_id = ?
                ORDER BY deployed_at DESC, deployment_id DESC LIMIT ?
                """,
                (clean_application_id, clean_limit),
            ).fetchall()
        return tuple(_deployment_from_row(row) for row in rows)

    def record_audit_event(self, principal: Principal, event: AuditEvent) -> AuditEvent:
        tenant_id = _principal_tenant(principal)
        _require_tenant_record(tenant_id, event.tenant_id)
        with self._write_lock, self._write_transaction() as connection:
            if event.project_id is not None:
                self._require_project(connection, tenant_id, event.project_id)
            if event.application_id is not None:
                self._require_application(connection, tenant_id, event.application_id)
            if event.revision_id is not None:
                if event.application_id is None or self._select_revision(
                    connection, tenant_id, event.application_id, event.revision_id
                ) is None:
                    raise ApplicationRevisionUnavailableError()
            self._insert_audit_event(connection, event)
        return event

    def list_audit_events(
        self, principal: Principal, *, application_id: str | None = None, limit: int = 50
    ) -> tuple[AuditEvent, ...]:
        tenant_id = _principal_tenant(principal)
        clean_limit = _validate_limit(limit)
        clean_application_id = (
            None if application_id is None else _safe_application_lookup_id(application_id)
        )
        with self._read_connection() as connection:
            if clean_application_id is not None:
                self._require_application(connection, tenant_id, clean_application_id)
                query = (
                    "SELECT * FROM application_audit_events WHERE tenant_id = ? "
                    "AND application_id = ? ORDER BY occurred_at DESC, rowid DESC LIMIT ?"
                )
                parameters: tuple[object, ...] = (tenant_id.value, clean_application_id, clean_limit)
            else:
                query = (
                    "SELECT * FROM application_audit_events WHERE tenant_id = ? "
                    "ORDER BY occurred_at DESC, rowid DESC LIMIT ?"
                )
                parameters = (tenant_id.value, clean_limit)
            rows = connection.execute(query, parameters).fetchall()
        return tuple(_audit_event_from_row(row) for row in rows)

    def save_evaluation(
        self, principal: Principal, report: ApplicationEvaluationReport
    ) -> ApplicationEvaluationReport:
        tenant_id = _principal_tenant(principal)
        if not isinstance(report, ApplicationEvaluationReport):
            raise ApplicationValidationError("evaluation report must use the typed contract.")
        encoded = report.to_json()
        if len(encoded.encode("utf-8")) > MAX_EVALUATION_REPORT_BYTES:
            raise ApplicationValidationError("evaluation report exceeds the size limit.")
        with self._write_lock, self._write_transaction() as connection:
            revision = self._select_revision(
                connection, tenant_id, report.application_id, report.revision_id
            )
            if revision is None or int(revision["revision_number"]) != report.revision_number:
                raise ApplicationRevisionUnavailableError()
            try:
                connection.execute(
                    """INSERT INTO application_evaluations (
                    revision_id, generated_at, configuration_digest, report_json
                    ) VALUES (?, ?, ?, ?)""",
                    (report.revision_id, report.generated_at, report.configuration_digest, encoded),
                )
            except sqlite3.IntegrityError as error:
                raise ApplicationStoreStorageError() from error
        return report

    def list_evaluations(
        self, principal: Principal, application_id: str, revision_id: str, *, limit: int = 50
    ) -> tuple[ApplicationEvaluationReport, ...]:
        tenant_id = _principal_tenant(principal)
        clean_application_id = _safe_application_lookup_id(application_id)
        clean_revision_id = _safe_revision_lookup_id(revision_id)
        clean_limit = _validate_limit(limit)
        with self._read_connection() as connection:
            if self._select_revision(connection, tenant_id, clean_application_id, clean_revision_id) is None:
                raise ApplicationRevisionUnavailableError()
            rows = connection.execute(
                """SELECT report_json FROM application_evaluations WHERE revision_id = ?
                ORDER BY generated_at DESC, rowid DESC LIMIT ?""",
                (clean_revision_id, clean_limit),
            ).fetchall()
        try:
            return tuple(ApplicationEvaluationReport.from_json(row["report_json"]) for row in rows)
        except ApplicationEvaluationError as error:
            raise ApplicationStoreSchemaError() from error

    def _initialize(self) -> None:
        with self._write_lock, self._write_transaction(validate_schema=False) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 0:
                if _existing_schema_tables(connection):
                    raise ApplicationStoreSchemaError()
                connection.executescript(_CREATE_SCHEMA_SQL)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif version == 1:
                self._validate_schema(
                    connection,
                    expected_columns={
                        table: columns
                        for table, columns in _EXPECTED_COLUMNS.items()
                        if table not in {"application_drafts", "application_evaluations"}
                    },
                )
                self._migrate_v1_to_v2(connection)
                version = 2
            if version == 2:
                self._migrate_v2_to_v3(connection)
                version = 3
            if version == 3:
                self._migrate_v3_to_v4(connection)
                version = 4
            if version == 4:
                self._migrate_v4_to_v5(connection)
                version = 5
            if version == 5:
                self._migrate_v5_to_v6(connection)
                version = 6
            if version == 6:
                self._migrate_v6_to_v7(connection)
            elif version not in {0, _SCHEMA_VERSION}:
                raise ApplicationStoreSchemaError()
            self._validate_schema(connection)

    @staticmethod
    def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
        """Add the explicit cloud-answer policy to every legacy configuration."""
        rows = connection.execute(
            "SELECT revision_id, configuration_json FROM application_revisions"
        ).fetchall()
        for row in rows:
            configuration = _decode_v1_configuration(row["configuration_json"])
            connection.execute(
                "UPDATE application_revisions SET configuration_json = ? WHERE revision_id = ?",
                (_encode_legacy_configuration(configuration), row["revision_id"]),
            )
        connection.execute("PRAGMA user_version = 2")

    @staticmethod
    def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
        connection.execute("ALTER TABLE application_audit_events RENAME TO application_audit_events_v2")
        connection.execute(_CREATE_AUDIT_EVENTS_SQL)
        connection.execute(
            """INSERT INTO application_audit_events SELECT audit_event_id, tenant_id, event_type,
            occurred_at, actor, summary, project_id, application_id, revision_id
            FROM application_audit_events_v2"""
        )
        connection.execute("DROP TABLE application_audit_events_v2")
        connection.execute(
            "CREATE INDEX idx_application_audit_events_tenant_occurred ON "
            "application_audit_events (tenant_id, occurred_at DESC, audit_event_id DESC)"
        )
        connection.execute("PRAGMA user_version = 3")

    @staticmethod
    def _migrate_v3_to_v4(connection: sqlite3.Connection) -> None:
        """Name the retrieval strategy in every immutable application configuration."""
        rows = connection.execute(
            "SELECT revision_id, configuration_json FROM application_revisions"
        ).fetchall()
        for row in rows:
            configuration = _decode_legacy_configuration(row["configuration_json"])
            connection.execute(
                "UPDATE application_revisions SET configuration_json = ? WHERE revision_id = ?",
                (_encode_v5_configuration(configuration), row["revision_id"]),
            )
        connection.execute("PRAGMA user_version = 4")

    @staticmethod
    def _migrate_v4_to_v5(connection: sqlite3.Connection) -> None:
        connection.execute("ALTER TABLE application_audit_events RENAME TO application_audit_events_v4")
        connection.execute(_CREATE_AUDIT_EVENTS_SQL)
        connection.execute(
            """INSERT INTO application_audit_events SELECT audit_event_id, tenant_id, event_type,
            occurred_at, actor, summary, project_id, application_id, revision_id
            FROM application_audit_events_v4"""
        )
        connection.execute("DROP TABLE application_audit_events_v4")
        connection.execute(
            "CREATE INDEX idx_application_audit_events_tenant_occurred ON "
            "application_audit_events (tenant_id, occurred_at DESC, audit_event_id DESC)"
        )
        connection.execute(_CREATE_DRAFTS_SQL)
        connection.execute(
            """INSERT INTO application_drafts (
            application_id, version, configuration_json, updated_at, updated_by, change_summary
            ) SELECT application_id, 0, NULL, updated_at, 'migration', '' FROM applications"""
        )
        connection.execute("PRAGMA user_version = 5")

    @staticmethod
    def _migrate_v5_to_v6(connection: sqlite3.Connection) -> None:
        """Add the deployment-managed model profile reference to every configuration."""

        rows = connection.execute(
            "SELECT revision_id, configuration_json FROM application_revisions"
        ).fetchall()
        for row in rows:
            configuration = _decode_v5_configuration(row["configuration_json"])
            connection.execute(
                "UPDATE application_revisions SET configuration_json = ? WHERE revision_id = ?",
                (_encode_configuration(configuration), row["revision_id"]),
            )
        drafts = connection.execute(
            "SELECT application_id, configuration_json FROM application_drafts "
            "WHERE configuration_json IS NOT NULL"
        ).fetchall()
        for draft in drafts:
            configuration = _decode_v5_configuration(draft["configuration_json"])
            connection.execute(
                "UPDATE application_drafts SET configuration_json = ? WHERE application_id = ?",
                (_encode_configuration(configuration), draft["application_id"]),
            )
        connection.execute("PRAGMA user_version = 6")

    @staticmethod
    def _migrate_v6_to_v7(connection: sqlite3.Connection) -> None:
        connection.execute(_CREATE_EVALUATIONS_SQL)
        connection.execute(
            "CREATE INDEX idx_application_evaluations_revision_generated ON "
            "application_evaluations (revision_id, generated_at DESC)"
        )
        connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    @staticmethod
    def _validate_schema(
        connection: sqlite3.Connection,
        *,
        expected_columns: dict[str, tuple[tuple[str, str, int, int], ...]] | None = None,
    ) -> None:
        expected = _EXPECTED_COLUMNS if expected_columns is None else expected_columns
        tables = _existing_schema_tables(connection)
        if tables != frozenset(expected):
            raise ApplicationStoreSchemaError()
        for table, expected_table_columns in expected.items():
            actual_table_columns = tuple(
                (row["name"], row["type"].upper(), int(row["notnull"]), int(row["pk"]))
                for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            )
            if actual_table_columns != expected_table_columns:
                raise ApplicationStoreSchemaError()
        indexes = {
            str(row["name"]): (int(row["unique"]), int(row["partial"]))
            for table in expected
            for row in connection.execute(f"PRAGMA index_list({table})").fetchall()
            if str(row["name"]).startswith("idx_")
        }
        expected_indexes = {
            name: definition
            for name, definition in _EXPECTED_CUSTOM_INDEXES.items()
            if name != "idx_application_evaluations_revision_generated"
            or "application_evaluations" in expected
        }
        if indexes != expected_indexes:
            raise ApplicationStoreSchemaError()
        expected_index_columns_by_name = {
            name: columns
            for name, columns in _EXPECTED_INDEX_COLUMNS.items()
            if name in expected_indexes
        }
        for name, expected_index_columns in expected_index_columns_by_name.items():
            actual_index_columns = tuple(
                str(row["name"])
                for row in connection.execute(f"PRAGMA index_info({name})").fetchall()
            )
            if actual_index_columns != expected_index_columns:
                raise ApplicationStoreSchemaError()

    def _require_project(
        self, connection: sqlite3.Connection, tenant_id: TenantId, project_id: str
    ) -> Project:
        row = connection.execute(
            "SELECT * FROM projects WHERE project_id = ? AND tenant_id = ?",
            (project_id, tenant_id.value),
        ).fetchone()
        if row is None:
            raise ProjectUnavailableError()
        return _project_from_row(row)

    @staticmethod
    def _insert_audit_event(connection: sqlite3.Connection, event: AuditEvent) -> None:
        try:
            connection.execute(
                """
                INSERT INTO application_audit_events (
                    audit_event_id, tenant_id, event_type, occurred_at, actor, summary,
                    project_id, application_id, revision_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.audit_event_id,
                    event.tenant_id.value,
                    event.event_type.value,
                    event.occurred_at,
                    event.actor,
                    event.summary,
                    event.project_id,
                    event.application_id,
                    event.revision_id,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ApplicationStoreStorageError() from error

    def _require_application(
        self, connection: sqlite3.Connection, tenant_id: TenantId, application_id: str
    ) -> Application:
        row = self._select_application(connection, tenant_id, application_id)
        if row is None:
            raise ApplicationUnavailableError()
        return _application_from_row(row)

    @staticmethod
    def _select_application(
        connection: sqlite3.Connection, tenant_id: TenantId, application_id: str
    ) -> sqlite3.Row | None:
        return cast(
            sqlite3.Row | None,
            connection.execute(
                "SELECT * FROM applications WHERE application_id = ? AND tenant_id = ?",
                (application_id, tenant_id.value),
            ).fetchone(),
        )

    @staticmethod
    def _select_revision(
        connection: sqlite3.Connection,
        tenant_id: TenantId,
        application_id: str,
        revision_id: str,
    ) -> sqlite3.Row | None:
        return cast(
            sqlite3.Row | None,
            connection.execute(
                """
                SELECT r.* FROM application_revisions AS r
                JOIN applications AS a ON a.application_id = r.application_id
                WHERE r.revision_id = ? AND r.application_id = ? AND a.tenant_id = ?
                """,
                (revision_id, application_id, tenant_id.value),
            ).fetchone(),
        )

    def _connect(self) -> sqlite3.Connection:
        return self._database.connect(ApplicationStoreStorageError)

    def _read_connection(self) -> AbstractContextManager[sqlite3.Connection]:
        return self._database.read(ApplicationStoreStorageError)

    def _write_transaction(
        self, *, validate_schema: bool = True
    ) -> AbstractContextManager[sqlite3.Connection]:
        return self._database.immediate_transaction(
            ApplicationStoreStorageError,
            pass_through=(ApplicationStoreError, ApplicationContractError),
            before_write=self._validate_schema if validate_schema else None,
        )


_CREATE_AUDIT_EVENTS_SQL = """
CREATE TABLE application_audit_events (
    audit_event_id TEXT PRIMARY KEY NOT NULL,
    tenant_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN (
        'project_created', 'application_created', 'application_archived', 'draft_updated',
        'revision_created', 'deployment_created'
    )),
    occurred_at REAL NOT NULL CHECK (occurred_at >= 0),
    actor TEXT NOT NULL,
    summary TEXT NOT NULL,
    project_id TEXT,
    application_id TEXT,
    revision_id TEXT
)
"""


_CREATE_DRAFTS_SQL = """
CREATE TABLE application_drafts (
    application_id TEXT PRIMARY KEY NOT NULL REFERENCES applications(application_id),
    version INTEGER NOT NULL CHECK (version >= 0),
    configuration_json TEXT,
    updated_at REAL NOT NULL CHECK (updated_at >= 0),
    updated_by TEXT NOT NULL,
    change_summary TEXT NOT NULL
)
"""


_CREATE_EVALUATIONS_SQL = """
CREATE TABLE application_evaluations (
    revision_id TEXT NOT NULL REFERENCES application_revisions(revision_id),
    generated_at REAL NOT NULL CHECK (generated_at >= 0),
    configuration_digest TEXT NOT NULL,
    report_json TEXT NOT NULL,
    UNIQUE (revision_id, generated_at, configuration_digest)
)
"""


_CREATE_SCHEMA_SQL = """
CREATE TABLE projects (
    project_id TEXT PRIMARY KEY NOT NULL,
    tenant_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at REAL NOT NULL CHECK (created_at >= 0),
    updated_at REAL NOT NULL CHECK (updated_at >= created_at)
);
CREATE TABLE applications (
    application_id TEXT PRIMARY KEY NOT NULL,
    tenant_id TEXT NOT NULL,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    display_name TEXT NOT NULL,
    application_kind TEXT NOT NULL CHECK (application_kind = 'knowledge_chat'),
    active_revision_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('active', 'archived')),
    created_at REAL NOT NULL CHECK (created_at >= 0),
    updated_at REAL NOT NULL CHECK (updated_at >= created_at)
);
CREATE TABLE application_revisions (
    revision_id TEXT PRIMARY KEY NOT NULL,
    application_id TEXT NOT NULL REFERENCES applications(application_id),
    revision_number INTEGER NOT NULL CHECK (revision_number >= 1),
    configuration_schema_version INTEGER NOT NULL CHECK (configuration_schema_version = 1),
    configuration_json TEXT NOT NULL,
    created_at REAL NOT NULL CHECK (created_at >= 0),
    created_by TEXT NOT NULL,
    change_summary TEXT NOT NULL,
    UNIQUE (application_id, revision_number)
);
CREATE TABLE application_drafts (
    application_id TEXT PRIMARY KEY NOT NULL REFERENCES applications(application_id),
    version INTEGER NOT NULL CHECK (version >= 0),
    configuration_json TEXT,
    updated_at REAL NOT NULL CHECK (updated_at >= 0),
    updated_by TEXT NOT NULL,
    change_summary TEXT NOT NULL
);
CREATE TABLE resource_bindings (
    binding_id TEXT PRIMARY KEY NOT NULL,
    application_id TEXT NOT NULL REFERENCES applications(application_id),
    revision_id TEXT NOT NULL REFERENCES application_revisions(revision_id),
    resource_kind TEXT NOT NULL CHECK (resource_kind = 'knowledge_base'),
    resource_id TEXT NOT NULL,
    access_mode TEXT NOT NULL CHECK (access_mode = 'read'),
    created_at REAL NOT NULL CHECK (created_at >= 0),
    UNIQUE (revision_id, resource_kind, resource_id)
);
CREATE TABLE deployments (
    deployment_id TEXT PRIMARY KEY NOT NULL,
    application_id TEXT NOT NULL REFERENCES applications(application_id),
    revision_id TEXT NOT NULL REFERENCES application_revisions(revision_id),
    environment TEXT NOT NULL CHECK (environment IN ('development', 'staging', 'production')),
    deployed_at REAL NOT NULL CHECK (deployed_at >= 0),
    deployed_by TEXT NOT NULL
);
CREATE TABLE application_audit_events (
    audit_event_id TEXT PRIMARY KEY NOT NULL,
    tenant_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN (
        'project_created', 'application_created', 'application_archived', 'draft_updated',
        'revision_created', 'deployment_created'
    )),
    occurred_at REAL NOT NULL CHECK (occurred_at >= 0),
    actor TEXT NOT NULL,
    summary TEXT NOT NULL,
    project_id TEXT,
    application_id TEXT,
    revision_id TEXT
);
CREATE TABLE application_evaluations (
    revision_id TEXT NOT NULL REFERENCES application_revisions(revision_id),
    generated_at REAL NOT NULL CHECK (generated_at >= 0),
    configuration_digest TEXT NOT NULL,
    report_json TEXT NOT NULL,
    UNIQUE (revision_id, generated_at, configuration_digest)
);
CREATE INDEX idx_projects_tenant_updated ON projects (tenant_id, updated_at DESC, project_id DESC);
CREATE INDEX idx_applications_tenant_project_updated
    ON applications (tenant_id, project_id, updated_at DESC, application_id DESC);
CREATE INDEX idx_revisions_application_number
    ON application_revisions (application_id, revision_number DESC);
CREATE INDEX idx_deployments_application_deployed
    ON deployments (application_id, deployed_at DESC, deployment_id DESC);
CREATE INDEX idx_application_audit_events_tenant_occurred
    ON application_audit_events (tenant_id, occurred_at DESC, audit_event_id DESC);
CREATE INDEX idx_application_evaluations_revision_generated
    ON application_evaluations (revision_id, generated_at DESC);
"""


_EXPECTED_COLUMNS = {
    "projects": (
        ("project_id", "TEXT", 1, 1), ("tenant_id", "TEXT", 1, 0),
        ("display_name", "TEXT", 1, 0), ("description", "TEXT", 1, 0),
        ("created_at", "REAL", 1, 0), ("updated_at", "REAL", 1, 0),
    ),
    "applications": (
        ("application_id", "TEXT", 1, 1), ("tenant_id", "TEXT", 1, 0),
        ("project_id", "TEXT", 1, 0), ("display_name", "TEXT", 1, 0),
        ("application_kind", "TEXT", 1, 0), ("active_revision_id", "TEXT", 0, 0),
        ("status", "TEXT", 1, 0), ("created_at", "REAL", 1, 0),
        ("updated_at", "REAL", 1, 0),
    ),
    "application_revisions": (
        ("revision_id", "TEXT", 1, 1), ("application_id", "TEXT", 1, 0),
        ("revision_number", "INTEGER", 1, 0), ("configuration_schema_version", "INTEGER", 1, 0),
        ("configuration_json", "TEXT", 1, 0), ("created_at", "REAL", 1, 0),
        ("created_by", "TEXT", 1, 0), ("change_summary", "TEXT", 1, 0),
    ),
    "application_drafts": (
        ("application_id", "TEXT", 1, 1), ("version", "INTEGER", 1, 0),
        ("configuration_json", "TEXT", 0, 0), ("updated_at", "REAL", 1, 0),
        ("updated_by", "TEXT", 1, 0), ("change_summary", "TEXT", 1, 0),
    ),
    "resource_bindings": (
        ("binding_id", "TEXT", 1, 1), ("application_id", "TEXT", 1, 0),
        ("revision_id", "TEXT", 1, 0), ("resource_kind", "TEXT", 1, 0),
        ("resource_id", "TEXT", 1, 0), ("access_mode", "TEXT", 1, 0),
        ("created_at", "REAL", 1, 0),
    ),
    "deployments": (
        ("deployment_id", "TEXT", 1, 1), ("application_id", "TEXT", 1, 0),
        ("revision_id", "TEXT", 1, 0), ("environment", "TEXT", 1, 0),
        ("deployed_at", "REAL", 1, 0), ("deployed_by", "TEXT", 1, 0),
    ),
    "application_audit_events": (
        ("audit_event_id", "TEXT", 1, 1), ("tenant_id", "TEXT", 1, 0),
        ("event_type", "TEXT", 1, 0), ("occurred_at", "REAL", 1, 0),
        ("actor", "TEXT", 1, 0), ("summary", "TEXT", 1, 0),
        ("project_id", "TEXT", 0, 0), ("application_id", "TEXT", 0, 0),
        ("revision_id", "TEXT", 0, 0),
    ),
    "application_evaluations": (
        ("revision_id", "TEXT", 1, 0), ("generated_at", "REAL", 1, 0),
        ("configuration_digest", "TEXT", 1, 0), ("report_json", "TEXT", 1, 0),
    ),
}

_EXPECTED_CUSTOM_INDEXES = {
    "idx_projects_tenant_updated": (0, 0),
    "idx_applications_tenant_project_updated": (0, 0),
    "idx_revisions_application_number": (0, 0),
    "idx_deployments_application_deployed": (0, 0),
    "idx_application_audit_events_tenant_occurred": (0, 0),
    "idx_application_evaluations_revision_generated": (0, 0),
}

_EXPECTED_INDEX_COLUMNS = {
    "idx_projects_tenant_updated": ("tenant_id", "updated_at", "project_id"),
    "idx_applications_tenant_project_updated": (
        "tenant_id", "project_id", "updated_at", "application_id"
    ),
    "idx_revisions_application_number": ("application_id", "revision_number"),
    "idx_deployments_application_deployed": ("application_id", "deployed_at", "deployment_id"),
    "idx_application_audit_events_tenant_occurred": (
        "tenant_id", "occurred_at", "audit_event_id"
    ),
    "idx_application_evaluations_revision_generated": ("revision_id", "generated_at"),
}


def _principal_tenant(principal: Principal) -> TenantId:
    if not isinstance(principal, Principal):
        raise TypeError("principal must be a Principal")
    return principal.tenant_id


def _require_tenant_record(expected: TenantId, actual: TenantId) -> None:
    if expected != actual:
        raise ApplicationUnavailableError()


def _validate_limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_LIST_LIMIT:
        raise ApplicationValidationError(f"limit must be between 1 and {_MAX_LIST_LIMIT}.")
    return value


def _safe_project_lookup_id(value: object) -> str:
    try:
        return validate_project_id(value)
    except ApplicationValidationError:
        raise ProjectUnavailableError() from None


def _safe_application_lookup_id(value: object) -> str:
    try:
        return validate_application_id(value)
    except ApplicationValidationError:
        raise ApplicationUnavailableError() from None


def _safe_revision_lookup_id(value: object) -> str:
    try:
        return validate_revision_id(value)
    except ApplicationValidationError:
        raise ApplicationRevisionUnavailableError() from None


def _safe_deployment_lookup_id(value: object) -> str:
    try:
        return validate_deployment_id(value)
    except ApplicationValidationError:
        raise DeploymentUnavailableError() from None


def _normalize_bindings(
    revision: ApplicationRevision, bindings: Sequence[ResourceBinding]
) -> tuple[ResourceBinding, ...]:
    if isinstance(bindings, (str, bytes)) or not isinstance(bindings, Sequence):
        raise ApplicationValidationError("bindings must be a sequence of ResourceBinding values.")
    normalized = tuple(bindings)
    if not normalized or any(not isinstance(binding, ResourceBinding) for binding in normalized):
        raise ApplicationValidationError("bindings must not be empty and must contain ResourceBinding values.")
    if any(
        binding.application_id != revision.application_id or binding.revision_id != revision.revision_id
        for binding in normalized
    ):
        raise ApplicationValidationError("Resource bindings must belong to the supplied revision.")
    if len({binding.binding_id for binding in normalized}) != len(normalized):
        raise ApplicationValidationError("Resource binding IDs must be unique.")
    return normalized


def _validate_revision_for_application(
    application: Application,
    revision: ApplicationRevision,
    bindings: tuple[ResourceBinding, ...],
) -> None:
    if application.application_kind is not ApplicationKind.KNOWLEDGE_CHAT:
        raise ApplicationValidationError("Unsupported application kind.")
    if not isinstance(revision.configuration, KnowledgeChatConfiguration):
        raise ApplicationValidationError("knowledge_chat requires a KnowledgeChatConfiguration.")
    knowledge_base_ids = {
        binding.resource_id
        for binding in bindings
        if binding.resource_kind is ResourceKind.KNOWLEDGE_BASE
        and binding.access_mode is ResourceAccessMode.READ
    }
    if knowledge_base_ids != set(revision.configuration.knowledge_base_ids):
        raise ApplicationValidationError(
            "Knowledge-chat resource bindings must exactly match configured knowledge bases."
        )


def _validate_publish_audit_event(
    tenant_id: TenantId, deployment: Deployment, audit_event: AuditEvent
) -> None:
    if audit_event.tenant_id != tenant_id:
        raise ApplicationUnavailableError()
    if audit_event.event_type is not ApplicationAuditEventType.DEPLOYMENT_CREATED:
        raise ApplicationValidationError("Publishing requires a deployment-created audit event.")
    if (
        audit_event.application_id != deployment.application_id
        or audit_event.revision_id != deployment.revision_id
    ):
        raise ApplicationValidationError("Deployment audit event does not match the deployment.")


def _encode_configuration(configuration: KnowledgeChatConfiguration) -> str:
    payload = {
        "answer_policy": {
            "allow_cloud": configuration.answer_policy.allow_cloud,
            "allow_research": configuration.answer_policy.allow_research,
            "allow_web": configuration.answer_policy.allow_web,
            "require_citations": configuration.answer_policy.require_citations,
        },
        "knowledge_base_ids": list(configuration.knowledge_base_ids),
        "model_profile_id": configuration.model_profile_id,
        "retrieval_profile": configuration.retrieval_profile.value,
        "session_policy": {
            "enabled": configuration.session_policy.enabled,
            "ttl_seconds": configuration.session_policy.ttl_seconds,
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if len(encoded.encode("utf-8")) > _MAX_CONFIGURATION_BYTES:
        raise ApplicationValidationError("Encoded application configuration exceeds the size limit.")
    return encoded


def _encode_v5_configuration(configuration: KnowledgeChatConfiguration) -> str:
    """Encode the exact schema-v5 shape during one-way historical migration."""
    payload = {
        "answer_policy": {
            "allow_cloud": configuration.answer_policy.allow_cloud,
            "allow_research": configuration.answer_policy.allow_research,
            "allow_web": configuration.answer_policy.allow_web,
            "require_citations": configuration.answer_policy.require_citations,
        },
        "knowledge_base_ids": list(configuration.knowledge_base_ids),
        "retrieval_profile": configuration.retrieval_profile.value,
        "session_policy": {
            "enabled": configuration.session_policy.enabled,
            "ttl_seconds": configuration.session_policy.ttl_seconds,
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _decode_configuration(value: object) -> KnowledgeChatConfiguration:
    if not isinstance(value, str) or len(value.encode("utf-8")) > _MAX_CONFIGURATION_BYTES:
        raise ApplicationStoreSchemaError()
    try:
        payload = json.loads(value, object_pairs_hook=_strict_json_object)
        if not isinstance(payload, dict) or set(payload) != {
            "knowledge_base_ids", "model_profile_id", "retrieval_profile", "answer_policy", "session_policy"
        }:
            raise ValueError
        answer = payload["answer_policy"]
        session = payload["session_policy"]
        if not isinstance(answer, dict) or set(answer) != {
            "require_citations", "allow_cloud", "allow_web", "allow_research"
        }:
            raise ValueError
        if not isinstance(session, dict) or set(session) != {"enabled", "ttl_seconds"}:
            raise ValueError
        return KnowledgeChatConfiguration(
            knowledge_base_ids=tuple(payload["knowledge_base_ids"]),
            model_profile_id=payload["model_profile_id"],
            retrieval_profile=RetrievalProfile(payload["retrieval_profile"]),
            answer_policy=AnswerPolicy(**answer),
            session_policy=SessionPolicy(**session),
        )
    except (ApplicationContractError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ApplicationStoreSchemaError() from error


def _decode_v1_configuration(value: object) -> KnowledgeChatConfiguration:
    """Decode the original configuration shape during the one-way store migration."""
    if not isinstance(value, str) or len(value.encode("utf-8")) > _MAX_CONFIGURATION_BYTES:
        raise ApplicationStoreSchemaError()
    try:
        payload = json.loads(value, object_pairs_hook=_strict_json_object)
        if not isinstance(payload, dict) or set(payload) != {
            "knowledge_base_ids", "answer_policy", "session_policy"
        }:
            raise ValueError
        answer = payload["answer_policy"]
        session = payload["session_policy"]
        if not isinstance(answer, dict) or set(answer) != {
            "require_citations", "allow_web", "allow_research"
        }:
            raise ValueError
        if not isinstance(session, dict) or set(session) != {"enabled", "ttl_seconds"}:
            raise ValueError
        return KnowledgeChatConfiguration(
            knowledge_base_ids=tuple(payload["knowledge_base_ids"]),
            answer_policy=AnswerPolicy(**answer),
            session_policy=SessionPolicy(**session),
        )
    except (ApplicationContractError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ApplicationStoreSchemaError() from error


def _encode_legacy_configuration(configuration: KnowledgeChatConfiguration) -> str:
    payload = {
        "answer_policy": {
            "allow_cloud": configuration.answer_policy.allow_cloud,
            "allow_research": configuration.answer_policy.allow_research,
            "allow_web": configuration.answer_policy.allow_web,
            "require_citations": configuration.answer_policy.require_citations,
        },
        "knowledge_base_ids": list(configuration.knowledge_base_ids),
        "session_policy": {
            "enabled": configuration.session_policy.enabled,
            "ttl_seconds": configuration.session_policy.ttl_seconds,
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _decode_legacy_configuration(value: object) -> KnowledgeChatConfiguration:
    """Decode the store-v2/v3 shape before retrieval profiles were explicit."""
    if not isinstance(value, str) or len(value.encode("utf-8")) > _MAX_CONFIGURATION_BYTES:
        raise ApplicationStoreSchemaError()
    try:
        payload = json.loads(value, object_pairs_hook=_strict_json_object)
        if not isinstance(payload, dict) or set(payload) != {
            "knowledge_base_ids", "answer_policy", "session_policy"
        }:
            raise ValueError
        answer = payload["answer_policy"]
        session = payload["session_policy"]
        if not isinstance(answer, dict) or set(answer) != {
            "require_citations", "allow_cloud", "allow_web", "allow_research"
        }:
            raise ValueError
        if not isinstance(session, dict) or set(session) != {"enabled", "ttl_seconds"}:
            raise ValueError
        return KnowledgeChatConfiguration(
            knowledge_base_ids=tuple(payload["knowledge_base_ids"]),
            answer_policy=AnswerPolicy(**answer),
            session_policy=SessionPolicy(**session),
        )
    except (ApplicationContractError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ApplicationStoreSchemaError() from error


def _decode_v5_configuration(value: object) -> KnowledgeChatConfiguration:
    """Decode schema-v5 configuration before model profiles became explicit."""
    if not isinstance(value, str) or len(value.encode("utf-8")) > _MAX_CONFIGURATION_BYTES:
        raise ApplicationStoreSchemaError()
    try:
        payload = json.loads(value, object_pairs_hook=_strict_json_object)
        if not isinstance(payload, dict) or set(payload) != {
            "knowledge_base_ids", "retrieval_profile", "answer_policy", "session_policy"
        }:
            raise ValueError
        answer = payload["answer_policy"]
        session = payload["session_policy"]
        if not isinstance(answer, dict) or set(answer) != {
            "require_citations", "allow_cloud", "allow_web", "allow_research"
        }:
            raise ValueError
        if not isinstance(session, dict) or set(session) != {"enabled", "ttl_seconds"}:
            raise ValueError
        return KnowledgeChatConfiguration(
            knowledge_base_ids=tuple(payload["knowledge_base_ids"]),
            model_profile_id=DEFAULT_MODEL_PROFILE_ID,
            retrieval_profile=RetrievalProfile(payload["retrieval_profile"]),
            answer_policy=AnswerPolicy(**answer),
            session_policy=SessionPolicy(**session),
        )
    except (ApplicationContractError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ApplicationStoreSchemaError() from error


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _project_from_row(row: sqlite3.Row) -> Project:
    try:
        return Project(
            project_id=row["project_id"], tenant_id=TenantId(row["tenant_id"]),
            display_name=row["display_name"], description=row["description"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )
    except (ApplicationContractError, KeyError, TypeError, ValueError) as error:
        raise ApplicationStoreSchemaError() from error


def _application_from_row(row: sqlite3.Row) -> Application:
    try:
        return Application(
            application_id=row["application_id"], tenant_id=TenantId(row["tenant_id"]),
            project_id=row["project_id"], display_name=row["display_name"],
            application_kind=ApplicationKind(row["application_kind"]),
            active_revision_id=row["active_revision_id"], status=ApplicationStatus(row["status"]),
            created_at=row["created_at"], updated_at=row["updated_at"],
        )
    except (ApplicationContractError, KeyError, TypeError, ValueError) as error:
        raise ApplicationStoreSchemaError() from error


def _revision_from_row(row: sqlite3.Row) -> ApplicationRevision:
    try:
        return ApplicationRevision(
            revision_id=row["revision_id"], application_id=row["application_id"],
            revision_number=row["revision_number"],
            configuration_schema_version=row["configuration_schema_version"],
            configuration=_decode_configuration(row["configuration_json"]),
            created_at=row["created_at"], created_by=row["created_by"],
            change_summary=row["change_summary"],
        )
    except (ApplicationContractError, KeyError, TypeError, ValueError) as error:
        raise ApplicationStoreSchemaError() from error


def _draft_from_row(row: sqlite3.Row) -> ApplicationDraft:
    try:
        raw_configuration = row["configuration_json"]
        return ApplicationDraft(
            application_id=row["application_id"],
            version=row["version"],
            configuration=(
                None if raw_configuration is None else _decode_configuration(raw_configuration)
            ),
            updated_at=row["updated_at"],
            updated_by=row["updated_by"],
            change_summary=row["change_summary"],
        )
    except (ApplicationContractError, KeyError, TypeError, ValueError) as error:
        raise ApplicationStoreSchemaError() from error


def _binding_from_row(row: sqlite3.Row) -> ResourceBinding:
    try:
        return ResourceBinding(
            binding_id=row["binding_id"], application_id=row["application_id"],
            revision_id=row["revision_id"], resource_kind=ResourceKind(row["resource_kind"]),
            resource_id=row["resource_id"], access_mode=ResourceAccessMode(row["access_mode"]),
            created_at=row["created_at"],
        )
    except (ApplicationContractError, KeyError, TypeError, ValueError) as error:
        raise ApplicationStoreSchemaError() from error


def _deployment_from_row(row: sqlite3.Row) -> Deployment:
    try:
        return Deployment(
            deployment_id=row["deployment_id"], application_id=row["application_id"],
            revision_id=row["revision_id"], environment=DeploymentEnvironment(row["environment"]),
            deployed_at=row["deployed_at"], deployed_by=row["deployed_by"],
        )
    except (ApplicationContractError, KeyError, TypeError, ValueError) as error:
        raise ApplicationStoreSchemaError() from error


def _audit_event_from_row(row: sqlite3.Row) -> AuditEvent:
    try:
        return AuditEvent(
            audit_event_id=row["audit_event_id"], tenant_id=TenantId(row["tenant_id"]),
            event_type=ApplicationAuditEventType(row["event_type"]), occurred_at=row["occurred_at"],
            actor=row["actor"], summary=row["summary"], project_id=row["project_id"],
            application_id=row["application_id"], revision_id=row["revision_id"],
        )
    except (ApplicationContractError, KeyError, TypeError, ValueError) as error:
        raise ApplicationStoreSchemaError() from error


def _existing_schema_tables(connection: sqlite3.Connection) -> frozenset[str]:
    return frozenset(
        str(row["name"])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    )


__all__ = [
    "ApplicationDraftConflictError",
    "ApplicationPublishConflictError",
    "ApplicationRevisionUnavailableError",
    "ApplicationStore",
    "ApplicationStoreError",
    "ApplicationStoreSchemaError",
    "ApplicationStoreStorageError",
    "ApplicationUnavailableError",
    "DeploymentUnavailableError",
    "ProjectUnavailableError",
]
