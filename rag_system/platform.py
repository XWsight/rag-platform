"""Stable application facade for tenant-scoped RAG use cases."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence

from rag_system.answer_workflow import KnowledgeBaseAnswerWorkflow
from rag_system.application import (
    IdempotencyInProgressError,
    KnowledgeBaseNotReadyError,
    KnowledgeBaseSubmission,
    PlatformError,
    PlatformIntegrityError,
    PlatformUnavailableError,
    PlatformValidationError,
    UploadDocument,
)
from rag_system.application_ports import (
    DocumentStore,
    IdempotencyRepository,
    JobExecutor,
    KnowledgeBaseRepository,
    KnowledgeService,
)
from rag_system.config import Settings
from rag_system.domain import AnswerRequest, AnswerResult
from rag_system.job_contracts import JobId, JobSnapshot
from rag_system.knowledge_base_contracts import KnowledgeBaseRecord
from rag_system.knowledge_base_lifecycle import KnowledgeBaseLifecycle
from rag_system.metrics import OperationalMetrics, create_operational_metrics
from rag_system.tenancy import Principal


class RagPlatform:
    """Expose stable use cases while delegating stateful workflows explicitly.

    The default composition remains one durable node: SQLite, uploaded files,
    and local vector-index files share one storage volume.  The facade does not
    own lifecycle state, so future execution or storage profiles can replace a
    workflow without changing delivery adapters.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        service: KnowledgeService,
        catalog: KnowledgeBaseRepository,
        file_store: DocumentStore,
        jobs: JobExecutor,
        idempotency: IdempotencyRepository,
        metrics: OperationalMetrics | None = None,
        document_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self.settings = settings.validate()
        self.service = service
        self.catalog = catalog
        self.file_store = file_store
        self.jobs = jobs
        self.idempotency = idempotency
        self.metrics = metrics or create_operational_metrics()
        self._lifecycle = KnowledgeBaseLifecycle(
            settings=self.settings,
            service=service,
            catalog=catalog,
            file_store=file_store,
            jobs=jobs,
            idempotency=idempotency,
            metrics=self.metrics,
            document_id_factory=document_id_factory,
        )
        self._answers = KnowledgeBaseAnswerWorkflow(
            settings=self.settings,
            catalog=catalog,
            service=service,
            assets=self._lifecycle.assets,
            resource_locks=self._lifecycle.resource_locks,
            metrics=self.metrics,
        )

    def create_knowledge_base(
        self,
        principal: Principal,
        *,
        display_name: str,
        documents: Sequence[UploadDocument],
        idempotency_key: str,
    ) -> KnowledgeBaseSubmission:
        return self._lifecycle.create(
            principal,
            display_name=display_name,
            documents=documents,
            idempotency_key=idempotency_key,
        )

    def get_knowledge_base(
        self,
        principal: Principal,
        resource_id: str,
    ) -> KnowledgeBaseRecord:
        return self._lifecycle.get(principal, resource_id)

    def list_knowledge_bases(
        self,
        principal: Principal,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[KnowledgeBaseRecord, ...]:
        return self._lifecycle.list(principal, limit=limit, offset=offset)

    def list_knowledge_bases_after(
        self,
        principal: Principal,
        *,
        updated_at: float,
        resource_id: str,
        limit: int = 50,
    ) -> tuple[KnowledgeBaseRecord, ...]:
        return self._lifecycle.list_after(
            principal,
            updated_at=updated_at,
            resource_id=resource_id,
            limit=limit,
        )

    def get_job(self, principal: Principal, job_id: JobId | str) -> JobSnapshot:
        return self._lifecycle.get_job(principal, job_id)

    def cancel_job(self, principal: Principal, job_id: JobId | str) -> JobSnapshot:
        return self._lifecycle.cancel_job(principal, job_id)

    def delete_knowledge_base(self, principal: Principal, resource_id: str) -> bool:
        return self._lifecycle.delete(principal, resource_id)

    def recover_incomplete(self, principals: Sequence[Principal]) -> int:
        return self._lifecycle.recover_incomplete(principals)

    def answer(
        self,
        principal: Principal,
        resource_id: str,
        request: AnswerRequest,
    ) -> AnswerResult:
        return self._answers.answer(principal, resource_id, request)

    def clear_session(
        self,
        principal: Principal,
        resource_id: str,
        session_id: str,
    ) -> bool:
        return self._answers.clear_session(principal, resource_id, session_id)

    def close(self) -> None:
        try:
            self.jobs.shutdown(wait=True, cancel_pending=True)
        finally:
            try:
                self.service.index_manager.close()
            finally:
                close_service = getattr(self.service, "close", None)
                if callable(close_service):
                    close_service()


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
