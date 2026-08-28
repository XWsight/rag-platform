"""Replaceable persistence and execution ports used by application workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from rag_system.application_contracts import (
    Application,
    ApplicationDraft,
    ApplicationRevision,
    AuditEvent,
    Deployment,
    Project,
    ResourceBinding,
)
from rag_system.application_evaluation import ApplicationEvaluationReport
from rag_system.knowledge_base_contracts import (
    DocumentManifest,
    KnowledgeBaseErrorCode,
    KnowledgeBaseRecord,
    KnowledgeBaseStatus,
)
from rag_system.domain import AnswerRequest, AnswerResult, IndexRef
from rag_system.idempotency import IdempotencyReservation
from rag_system.ingestion import IngestionResult
from rag_system.job_contracts import CancellationToken, JobId, JobRuntimeSnapshot, JobSnapshot
from rag_system.ports import Retriever
from rag_system.tenancy import Principal


class ApplicationRepository(Protocol):
    """Persistence port for tenant-scoped, versioned applications."""

    def create_project(self, principal: Principal, project: Project) -> Project: ...

    def get_project(self, principal: Principal, project_id: str) -> Project: ...

    def list_projects(self, principal: Principal, *, limit: int = 50) -> tuple[Project, ...]: ...

    def create_application(self, principal: Principal, application: Application) -> Application: ...

    def get_application(self, principal: Principal, application_id: str) -> Application: ...

    def list_applications(
        self, principal: Principal, project_id: str, *, limit: int = 50
    ) -> tuple[Application, ...]: ...

    def archive_application(
        self, principal: Principal, application_id: str, audit_event: AuditEvent, *, updated_at: float
    ) -> Application: ...

    def get_draft(self, principal: Principal, application_id: str) -> ApplicationDraft: ...

    def update_draft(
        self,
        principal: Principal,
        draft: ApplicationDraft,
        audit_event: AuditEvent,
        *,
        expected_version: int,
    ) -> ApplicationDraft: ...

    def create_revision(
        self,
        principal: Principal,
        revision: ApplicationRevision,
        bindings: Sequence[ResourceBinding],
    ) -> ApplicationRevision: ...

    def get_revision(
        self, principal: Principal, application_id: str, revision_id: str
    ) -> ApplicationRevision: ...

    def list_revisions(
        self, principal: Principal, application_id: str, *, limit: int = 50
    ) -> tuple[ApplicationRevision, ...]: ...

    def list_bindings(
        self, principal: Principal, application_id: str, revision_id: str
    ) -> tuple[ResourceBinding, ...]: ...

    def create_deployment(self, principal: Principal, deployment: Deployment) -> Deployment: ...

    def publish(
        self,
        principal: Principal,
        deployment: Deployment,
        audit_event: AuditEvent,
        *,
        updated_at: float,
        expected_active_revision_id: str | None | object = ...,
    ) -> Application: ...

    def get_deployment(self, principal: Principal, deployment_id: str) -> Deployment: ...

    def list_deployments(
        self, principal: Principal, application_id: str, *, limit: int = 50
    ) -> tuple[Deployment, ...]: ...

    def record_audit_event(self, principal: Principal, event: AuditEvent) -> AuditEvent: ...

    def list_audit_events(
        self, principal: Principal, *, application_id: str | None = None, limit: int = 50
    ) -> tuple[AuditEvent, ...]: ...

    def save_evaluation(
        self, principal: Principal, report: ApplicationEvaluationReport
    ) -> ApplicationEvaluationReport: ...

    def list_evaluations(
        self, principal: Principal, application_id: str, revision_id: str, *, limit: int = 50
    ) -> tuple[ApplicationEvaluationReport, ...]: ...


class KnowledgeBaseRepository(Protocol):
    def create(
        self,
        principal: Principal,
        display_name: str,
        *,
        idempotency_reservation_id: str | None = None,
    ) -> KnowledgeBaseRecord: ...

    def get(self, principal: Principal, resource_id: str) -> KnowledgeBaseRecord: ...

    def find_by_idempotency_reservation(
        self,
        principal: Principal,
        reservation_id: str,
    ) -> KnowledgeBaseRecord | None: ...

    def list(
        self,
        principal: Principal,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[KnowledgeBaseRecord, ...]: ...

    def list_after(
        self,
        principal: Principal,
        *,
        updated_at: float,
        resource_id: str,
        limit: int = 50,
    ) -> tuple[KnowledgeBaseRecord, ...]: ...

    def attach_manifest(
        self,
        principal: Principal,
        resource_id: str,
        documents: Sequence[DocumentManifest],
    ) -> KnowledgeBaseRecord: ...

    def transition(
        self,
        principal: Principal,
        resource_id: str,
        target: KnowledgeBaseStatus,
        *,
        internal_index_id: str | None = None,
        chunk_count: int | None = None,
        error_code: KnowledgeBaseErrorCode | None = None,
    ) -> KnowledgeBaseRecord: ...

    def delete(
        self,
        principal: Principal,
        resource_id: str,
    ) -> tuple[DocumentManifest, ...]: ...


class IdempotencyRepository(Protocol):
    def reserve(
        self,
        principal: Principal,
        operation: str,
        key: str,
        request_digest: str,
    ) -> IdempotencyReservation: ...

    def bind_result(
        self,
        principal: Principal,
        reservation_id: str,
        resource_id: str,
        job_id: str,
    ) -> IdempotencyReservation: ...

    def recover_binding(
        self,
        principal: Principal,
        reservation_id: str,
        resource_id: str,
        job_id: str,
    ) -> IdempotencyReservation: ...

    def abandon(self, principal: Principal, reservation_id: str) -> bool: ...

    def purge_expired(self) -> int: ...


JobTask = Callable[[CancellationToken], Mapping[str, Any]]


class JobExecutor(Protocol):
    def submit(
        self,
        tenant_id: str,
        task: JobTask,
        *,
        idempotency_key: str,
    ) -> JobId: ...

    def get(self, tenant_id: str, job_id: JobId | str) -> JobSnapshot: ...

    def cancel(self, tenant_id: str, job_id: JobId | str) -> JobSnapshot: ...

    def operational_snapshot(self) -> JobRuntimeSnapshot: ...

    def shutdown(self, *, wait: bool = True, cancel_pending: bool = True) -> None: ...

    def healthcheck(self) -> bool: ...


class StoredAsset(Protocol):
    @property
    def relative_path(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def sha256(self) -> str: ...


class DocumentStore(Protocol):
    @property
    def root(self) -> Path: ...

    def planned_relative_path(
        self,
        tenant_id: str,
        resource_id: str,
        display_name: str,
    ) -> str: ...

    def healthcheck(self) -> bool: ...

    def save(
        self,
        tenant_id: str,
        resource_id: str,
        display_name: str,
        source: bytes | bytearray | memoryview | BinaryIO,
    ) -> StoredAsset: ...

    def resolve(self, tenant_id: str, resource_id: str) -> Path: ...

    def delete(self, tenant_id: str, resource_id: str) -> bool: ...


class IndexLifecycle(Protocol):
    def get(self, index_id: str) -> Retriever: ...

    def delete(self, index_id: str) -> bool: ...

    def close(self) -> None: ...

    def healthcheck(self) -> bool: ...


class KnowledgeService(Protocol):
    @property
    def index_manager(self) -> IndexLifecycle: ...

    def create_index(
        self,
        paths: Sequence[str] | None = None,
        *,
        namespace: str = "",
    ) -> IndexRef: ...

    def prepare_index(
        self,
        paths: Sequence[str] | None = None,
        *,
        namespace: str = "",
    ) -> IngestionResult: ...

    def create_prepared_index(self, ingestion: IngestionResult) -> IndexRef: ...

    def clear_session(self, session_id: str) -> bool: ...

    def answer(self, index_id: str, request: AnswerRequest) -> AnswerResult: ...

    def answer_many(self, index_ids: Sequence[str], request: AnswerRequest) -> AnswerResult: ...

    def close(self) -> None: ...


__all__ = [
    "ApplicationRepository",
    "DocumentStore",
    "IdempotencyRepository",
    "IndexLifecycle",
    "JobExecutor",
    "JobTask",
    "KnowledgeBaseRepository",
    "KnowledgeService",
    "StoredAsset",
]
