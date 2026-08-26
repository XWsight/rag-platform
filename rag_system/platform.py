"""Production application service for tenant-scoped knowledge bases."""

from __future__ import annotations

import hashlib
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

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
from rag_system.assets import AssetStoreFailure, KnowledgeBaseAssets
from rag_system.catalog import (
    KnowledgeBaseErrorCode,
    KnowledgeBaseRecord,
    KnowledgeBaseStatus,
)
from rag_system.config import Settings
from rag_system.coordination import ResourceJobRegistry, ResourceLockPool
from rag_system.domain import AnswerRequest, AnswerResult
from rag_system.file_store import FileStoreError, FileStoreIOError
from rag_system.idempotency import (
    IdempotencyConflictError,
    IdempotencyUnavailableError,
)
from rag_system.job_contracts import (
    CancellationToken,
    JobError,
    JobId,
    JobNotFoundError,
    JobSnapshot,
    JobStatus,
)
from rag_system.indexing import KnowledgeBaseIndexer
from rag_system.metrics import OperationalMetrics, create_operational_metrics
from rag_system.submission import UploadBatchPreparer
from rag_system.tenancy import Principal


class RagPlatform:
    """Coordinate catalog, file storage, jobs, indexes, and answering.

    The platform is deployable as one durable node: SQLite, uploaded files,
    and local vector-index files share one storage volume. Expensive indexing runs
    outside the request thread and all resource operations are tenant-scoped.
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
        self._assets = KnowledgeBaseAssets(file_store)
        self._indexing = KnowledgeBaseIndexer(
            service=service,
            catalog=catalog,
            assets=self._assets,
            metrics=self.metrics,
        )
        self._uploads = UploadBatchPreparer(
            max_file_bytes=self.settings.max_file_bytes,
            max_total_bytes=self.settings.max_total_bytes,
            max_documents=self.settings.max_documents,
            document_id_factory=document_id_factory,
        )
        self._resource_locks = ResourceLockPool(slots=64)
        self._jobs_by_resource = ResourceJobRegistry()
        self._answer_slots = threading.BoundedSemaphore(
            self.settings.max_concurrent_answers
        )

    def create_knowledge_base(
        self,
        principal: Principal,
        *,
        display_name: str,
        documents: Sequence[UploadDocument],
        idempotency_key: str,
    ) -> KnowledgeBaseSubmission:
        uploads = self._uploads.prepare(documents)
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise PlatformValidationError("idempotency key is required")

        request_digest = self._uploads.request_digest(display_name, uploads)
        reservation = self.idempotency.reserve(
            principal,
            "knowledge_base.create",
            idempotency_key.strip(),
            request_digest,
        )
        if not reservation.created:
            if not reservation.is_bound:
                record = self.catalog.find_by_idempotency_reservation(
                    principal,
                    reservation.reservation_id,
                )
                if record is None:
                    raise IdempotencyInProgressError(
                        "matching request is still in progress"
                    )
                current_job = self._job_for_resource(principal, record.resource_id)
                if (
                    record.status is KnowledgeBaseStatus.CANCELLING
                    and self._job_is_terminal(principal, current_job)
                ):
                    record = self._indexing.converge_cancel_intent(principal, record)
                    current_job = None
                if current_job is None and record.status in {
                    KnowledgeBaseStatus.READY,
                    KnowledgeBaseStatus.FAILED,
                }:
                    current_job = self._submit_status_job(principal, record)
                if current_job is None:
                    raise IdempotencyInProgressError(
                        "matching request recovery is still in progress"
                    )
                self._recover_idempotency_binding(principal, record, current_job)
                return KnowledgeBaseSubmission(record, current_job, replayed=True)
            bound_resource_id = reservation.resource_id
            bound_job_id = reservation.job_id
            if bound_resource_id is None or bound_job_id is None:
                raise PlatformIntegrityError("bound idempotency result is incomplete")
            record = self.catalog.get(principal, bound_resource_id)
            current_job = self._job_for_resource(principal, record.resource_id)
            if current_job is None:
                try:
                    bound_snapshot = self.jobs.get(
                        principal.tenant_id.value,
                        JobId(bound_job_id),
                    )
                    if (
                        record.status is KnowledgeBaseStatus.READY
                        and bound_snapshot.status is not JobStatus.SUCCEEDED
                    ):
                        current_job = None
                    else:
                        current_job = JobId(bound_job_id)
                except JobNotFoundError:
                    current_job = None
            if (
                record.status is KnowledgeBaseStatus.CANCELLING
                and self._job_is_terminal(principal, current_job)
            ):
                record = self._indexing.converge_cancel_intent(principal, record)
                current_job = None
            if current_job is None and record.status in {
                KnowledgeBaseStatus.READY,
                KnowledgeBaseStatus.FAILED,
            }:
                current_job = self._submit_status_job(principal, record)
                self._recover_idempotency_binding(
                    principal,
                    record,
                    current_job,
                )
            if current_job is None:
                raise IdempotencyInProgressError(
                    "matching request recovery is still in progress"
                )
            return KnowledgeBaseSubmission(
                record,
                current_job,
                replayed=True,
            )

        created_record: KnowledgeBaseRecord | None = None
        job_id: JobId | None = None
        stored_document_ids: frozenset[str] | None = None
        try:
            created_record = self.catalog.create(
                principal,
                display_name,
                idempotency_reservation_id=reservation.reservation_id,
            )
            planned = self._assets.plan(
                principal,
                uploads,
                new_document_id=self._uploads.new_document_id,
            )
            created_record = self.catalog.attach_manifest(
                principal,
                created_record.resource_id,
                tuple(item.manifest for item in planned),
            )
            try:
                self._assets.store(principal, planned)
            except AssetStoreFailure as failure:
                stored_document_ids = frozenset(
                    document.resource_id for document in failure.stored_documents
                )
                if isinstance(failure.__cause__, FileStoreIOError):
                    stored_document_ids |= frozenset(
                        document.resource_id for document in failure.attempted_documents
                    )
                if failure.__cause__ is None:
                    raise PlatformUnavailableError(
                        "document assets could not be stored"
                    ) from failure
                raise failure.__cause__ from None
            created_record = self.catalog.transition(
                principal,
                created_record.resource_id,
                KnowledgeBaseStatus.PENDING,
            )
            job_id = self._submit_indexing(
                principal,
                created_record.resource_id,
                idempotency_key=reservation.reservation_id,
            )
            self.idempotency.bind_result(
                principal,
                reservation.reservation_id,
                created_record.resource_id,
                job_id.value,
            )
        except Exception:
            cleanup_complete = created_record is None
            if created_record is not None:
                if job_id is None:
                    cleanup_complete = self._rollback_create(
                        principal,
                        created_record,
                        stored_document_ids=stored_document_ids,
                    )
                else:
                    try:
                        self.delete_knowledge_base(principal, created_record.resource_id)
                        cleanup_complete = True
                    except Exception:
                        pass
            if cleanup_complete:
                try:
                    self.idempotency.abandon(principal, reservation.reservation_id)
                except Exception:
                    pass
            raise
        if created_record is None or job_id is None:
            raise PlatformUnavailableError("knowledge base submission did not complete")
        return KnowledgeBaseSubmission(created_record, job_id, replayed=False)

    def get_knowledge_base(
        self,
        principal: Principal,
        resource_id: str,
    ) -> KnowledgeBaseRecord:
        return self.catalog.get(principal, resource_id)

    def list_knowledge_bases(
        self,
        principal: Principal,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[KnowledgeBaseRecord, ...]:
        return self.catalog.list(principal, limit=limit, offset=offset)

    def list_knowledge_bases_after(
        self,
        principal: Principal,
        *,
        updated_at: float,
        resource_id: str,
        limit: int = 50,
    ) -> tuple[KnowledgeBaseRecord, ...]:
        return self.catalog.list_after(
            principal,
            updated_at=updated_at,
            resource_id=resource_id,
            limit=limit,
        )

    def get_job(self, principal: Principal, job_id: JobId | str) -> JobSnapshot:
        return self.jobs.get(principal.tenant_id.value, job_id)

    def cancel_job(self, principal: Principal, job_id: JobId | str) -> JobSnapshot:
        snapshot = self.jobs.get(principal.tenant_id.value, job_id)
        resource_id = self._resource_for_job(principal, snapshot.job_id)
        if resource_id is not None:
            self._indexing.persist_cancel_intent(principal, resource_id)
        snapshot = self.jobs.cancel(principal.tenant_id.value, snapshot.job_id)
        if resource_id is not None:
            self._indexing.mark_failed(
                principal,
                resource_id,
                KnowledgeBaseErrorCode.INDEX_CANCELLED,
            )
        return snapshot

    def answer(
        self,
        principal: Principal,
        resource_id: str,
        request: AnswerRequest,
    ) -> AnswerResult:
        if not self._answer_slots.acquire(blocking=False):
            raise PlatformUnavailableError("answer capacity is temporarily exhausted")
        try:
            return self._answer(principal, resource_id, request)
        finally:
            self._answer_slots.release()

    def _answer(
        self,
        principal: Principal,
        resource_id: str,
        request: AnswerRequest,
    ) -> AnswerResult:
        record = self.catalog.get(principal, resource_id)
        if record.status is not KnowledgeBaseStatus.READY or not record.internal_index_id:
            raise KnowledgeBaseNotReadyError("knowledge base is not ready")

        try:
            self.service.index_manager.get(record.internal_index_id)
        except KeyError:
            with self._resource_locks.hold(resource_id):
                record = self.catalog.get(principal, resource_id)
                if record.status is not KnowledgeBaseStatus.READY or not record.internal_index_id:
                    raise KnowledgeBaseNotReadyError("knowledge base is not ready") from None
                paths = self._assets.resolve(principal, record)
                restored = self.service.create_index(
                    [str(path) for path in paths],
                    namespace=self._namespace(principal, resource_id),
                )
                if restored.index_id != record.internal_index_id:
                    self.service.index_manager.delete(restored.index_id)
                    raise PlatformIntegrityError(
                        "stored index identity does not match its catalog"
                    ) from None

        scoped_request = replace(
            request,
            session_id=self._session_id(principal, resource_id, request.session_id),
        )
        internal_index_id = record.internal_index_id
        if internal_index_id is None:
            raise KnowledgeBaseNotReadyError("knowledge base is not ready")
        try:
            result = self.service.answer(internal_index_id, scoped_request)
            self._record_external_call_failures(result)
            return result
        except KeyError:
            current = self.catalog.get(principal, resource_id)
            if current.status is not KnowledgeBaseStatus.READY:
                raise KnowledgeBaseNotReadyError("knowledge base is not ready") from None
            raise KnowledgeBaseNotReadyError("knowledge base index is being reloaded") from None

    def _record_external_call_failures(self, result: AnswerResult) -> None:
        """Publish bounded provider failures without exposing request content."""

        diagnostics = result.diagnostics
        self._record_external_call_failure(
            diagnostics.get("embedding_error"),
            provider="embedding",
            operation="embed",
        )
        self._record_external_call_failure(
            diagnostics.get("provider_error"),
            provider="chat",
            operation="generate",
        )
        planning_error = diagnostics.get("planning_error")
        if planning_error != "planner_unavailable":
            self._record_external_call_failure(
                planning_error,
                provider="chat",
                operation="plan",
            )
        web_errors = self._parse_error_counts(diagnostics.get("web_error_counts"))
        if web_errors:
            for error_name, count in web_errors:
                self._record_external_call_failure(
                    error_name,
                    provider="web_search",
                    operation="search",
                    count=count,
                )
        else:
            self._record_external_call_failure(
                diagnostics.get("web_error"),
                provider="web_search",
                operation="search",
                count=diagnostics.get("web_error_count"),
            )

    def _parse_error_counts(self, encoded: object) -> tuple[tuple[str, int], ...]:
        """Parse bounded service diagnostics without trusting arbitrary adapters."""

        if not isinstance(encoded, str) or not encoded or len(encoded) > 256:
            return ()
        remaining = self.settings.research_max_web_queries
        parsed: list[tuple[str, int]] = []
        for item in encoded.split(","):
            error_name, separator, raw_count = item.partition(":")
            if not separator or not error_name or not raw_count.isdecimal():
                return ()
            count = int(raw_count)
            if count < 1:
                return ()
            bounded_count = min(count, remaining)
            parsed.append((error_name, bounded_count))
            remaining -= bounded_count
            if remaining == 0:
                break
        return tuple(parsed)

    def _record_external_call_failure(
        self,
        error_name: object,
        *,
        provider: str,
        operation: str,
        count: object = 1,
    ) -> None:
        if not isinstance(error_name, str) or not error_name:
            return
        error_type = {
            "ProviderAuthenticationError": "authentication",
            "ProviderProtocolError": "protocol",
            "GroundingContractError": "protocol",
            "ProviderRateLimitError": "rate_limit",
            "TimeoutError": "timeout",
            "ProviderUnavailableError": "unavailable",
        }.get(error_name, "unknown")
        bounded_count = count if isinstance(count, int) and not isinstance(count, bool) else 1
        bounded_count = min(max(bounded_count, 1), self.settings.research_max_web_queries)
        try:
            self.metrics.external_call_errors_total.increment(
                amount=bounded_count,
                labels={
                    "provider": provider,
                    "operation": operation,
                    "error_type": error_type,
                },
            )
        except Exception:
            return

    def clear_session(
        self,
        principal: Principal,
        resource_id: str,
        session_id: str,
    ) -> bool:
        self.catalog.get(principal, resource_id)
        return self.service.clear_session(self._session_id(principal, resource_id, session_id))

    def delete_knowledge_base(self, principal: Principal, resource_id: str) -> bool:
        job_id = self._job_for_resource(principal, resource_id)
        if job_id is not None:
            try:
                self.jobs.cancel(principal.tenant_id.value, job_id)
            except JobError:
                pass

        with self._resource_locks.hold(resource_id):
            record = self.catalog.get(principal, resource_id)
            if record.status is not KnowledgeBaseStatus.DELETING:
                record = self.catalog.transition(
                    principal,
                    resource_id,
                    KnowledgeBaseStatus.DELETING,
                )
            if record.internal_index_id:
                self.service.index_manager.delete(record.internal_index_id)
            self._assets.delete(principal, record)
            self.catalog.delete(principal, resource_id)
            self._jobs_by_resource.unbind_resource(
                principal.tenant_id.value,
                resource_id,
            )
            return True

    def recover_incomplete(self, principals: Sequence[Principal]) -> int:
        """Resubmit durable pending/indexing records after process restart."""

        recovered = 0
        seen_tenants: set[str] = set()
        for principal in principals:
            tenant = principal.tenant_id.value
            if tenant in seen_tenants:
                continue
            seen_tenants.add(tenant)
            tenant_records: list[KnowledgeBaseRecord] = []
            offset = 0
            while True:
                records = self.catalog.list(principal, limit=100, offset=offset)
                tenant_records.extend(records)
                if len(records) < 100:
                    break
                offset += len(records)
            for record in tenant_records:
                if record.status is KnowledgeBaseStatus.DELETING:
                    self.delete_knowledge_base(principal, record.resource_id)
                    self._abandon_reservation_if_unbound(principal, record)
                    recovered += 1
                elif record.status is KnowledgeBaseStatus.PREPARING:
                    self._recover_preparing(principal, record)
                    recovered += 1
                elif record.status is KnowledgeBaseStatus.CANCELLING:
                    self.catalog.transition(
                        principal,
                        record.resource_id,
                        KnowledgeBaseStatus.FAILED,
                        error_code=KnowledgeBaseErrorCode.INDEX_CANCELLED,
                    )
                    recovered += 1
                elif record.status in {
                    KnowledgeBaseStatus.PENDING,
                    KnowledgeBaseStatus.INDEXING,
                }:
                    job_id = self._submit_indexing(
                        principal,
                        record.resource_id,
                        idempotency_key=f"recovery:{record.resource_id}:{record.version}",
                    )
                    self._recover_idempotency_binding(principal, record, job_id)
                    recovered += 1
        return recovered

    def _submit_status_job(
        self,
        principal: Principal,
        record: KnowledgeBaseRecord,
    ) -> JobId:
        def task(token: CancellationToken) -> dict[str, str | int]:
            token.raise_if_cancelled()
            return {
                "knowledge_base_id": record.resource_id,
                "status": record.status.value,
                "document_count": record.document_count,
                "chunk_count": record.chunk_count,
            }

        job_id = self.jobs.submit(
            principal.tenant_id.value,
            task,
            idempotency_key=f"recovered-state:{record.resource_id}:{record.version}",
        )
        self._jobs_by_resource.bind(
            principal.tenant_id.value,
            record.resource_id,
            job_id,
        )
        return job_id

    def _recover_idempotency_binding(
        self,
        principal: Principal,
        record: KnowledgeBaseRecord,
        job_id: JobId,
    ) -> None:
        reservation_id = record.idempotency_reservation_id
        if reservation_id is None:
            return
        try:
            self.idempotency.recover_binding(
                principal,
                reservation_id,
                record.resource_id,
                job_id.value,
            )
        except IdempotencyUnavailableError:
            return
        except IdempotencyConflictError:
            raise PlatformIntegrityError(
                "idempotency binding points to a different durable resource"
            ) from None

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

    def _submit_indexing(
        self,
        principal: Principal,
        resource_id: str,
        *,
        idempotency_key: str,
    ) -> JobId:
        def task(token: CancellationToken) -> Mapping[str, Any]:
            with self._resource_locks.hold(resource_id):
                return self._indexing.run(
                    principal,
                    resource_id,
                    token,
                    namespace=self._namespace(principal, resource_id),
                )

        job_id = self.jobs.submit(
            principal.tenant_id.value,
            task,
            idempotency_key=idempotency_key,
        )
        self._jobs_by_resource.bind(
            principal.tenant_id.value,
            resource_id,
            job_id,
        )
        return job_id

    def _rollback_create(
        self,
        principal: Principal,
        record: KnowledgeBaseRecord,
        *,
        stored_document_ids: frozenset[str] | None = None,
    ) -> bool:
        try:
            current = self.catalog.get(principal, record.resource_id)
            if current.status is not KnowledgeBaseStatus.DELETING:
                current = self.catalog.transition(
                    principal,
                    record.resource_id,
                    KnowledgeBaseStatus.DELETING,
                )
        except Exception:
            return False

        cleanup_succeeded = True
        for document in current.documents:
            document_id = self._assets.document_resource_id(principal, document)
            if stored_document_ids is not None and document_id not in stored_document_ids:
                continue
            try:
                self.file_store.delete(principal.tenant_id.value, document_id)
            except FileStoreError:
                cleanup_succeeded = False
        if cleanup_succeeded:
            try:
                self.catalog.delete(principal, record.resource_id)
            except Exception:
                return False
            return True
        return False

    def _recover_preparing(
        self,
        principal: Principal,
        record: KnowledgeBaseRecord,
    ) -> None:
        """Promote a fully stored upload or roll back an interrupted one."""

        try:
            self._assets.resolve(principal, record)
        except (FileStoreError, PlatformIntegrityError):
            if self._rollback_create(principal, record):
                self._abandon_reservation_if_unbound(principal, record)
            return

        pending = self.catalog.transition(
            principal,
            record.resource_id,
            KnowledgeBaseStatus.PENDING,
        )
        job_id = self._submit_indexing(
            principal,
            pending.resource_id,
            idempotency_key=f"recovery:{pending.resource_id}:{pending.version}",
        )
        self._recover_idempotency_binding(principal, pending, job_id)

    def _abandon_reservation_if_unbound(
        self,
        principal: Principal,
        record: KnowledgeBaseRecord,
    ) -> None:
        reservation_id = record.idempotency_reservation_id
        if reservation_id is None:
            return
        try:
            self.idempotency.abandon(principal, reservation_id)
        except IdempotencyConflictError:
            return

    @staticmethod
    def _namespace(principal: Principal, resource_id: str) -> str:
        return f"{principal.tenant_id.value}:{resource_id}"

    @staticmethod
    def _session_id(principal: Principal, resource_id: str, session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise PlatformValidationError("session_id is required")
        if len(session_id) > 128:
            raise PlatformValidationError("session_id is too long")
        identity = f"{principal.tenant_id.value}\0{resource_id}\0{session_id.strip()}"
        return "session_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _job_for_resource(self, principal: Principal, resource_id: str) -> JobId | None:
        tenant_id = principal.tenant_id.value
        job_id = self._jobs_by_resource.job_for(tenant_id, resource_id)
        if job_id is None:
            return None
        try:
            self.jobs.get(tenant_id, job_id)
        except JobNotFoundError:
            self._jobs_by_resource.unbind_resource(
                tenant_id,
                resource_id,
                expected_job_id=job_id,
            )
            return None
        return job_id

    def _job_is_terminal(self, principal: Principal, job_id: JobId | None) -> bool:
        if job_id is None:
            return True
        try:
            return self.jobs.get(principal.tenant_id.value, job_id).status.terminal
        except JobNotFoundError:
            return True

    def _resource_for_job(self, principal: Principal, job_id: JobId) -> str | None:
        return self._jobs_by_resource.resource_for(principal.tenant_id.value, job_id)


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
