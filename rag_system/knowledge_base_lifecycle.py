"""Durable knowledge-base lifecycle workflow for the application core."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from rag_system.application import (
    IdempotencyInProgressError,
    KnowledgeBaseSubmission,
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
from rag_system.file_store import FileStoreError, FileStoreIOError
from rag_system.idempotency import (
    IdempotencyConflictError,
    IdempotencyReservation,
    IdempotencyUnavailableError,
)
from rag_system.indexing import KnowledgeBaseIndexer
from rag_system.job_contracts import (
    CancellationToken,
    JobError,
    JobId,
    JobNotFoundError,
    JobSnapshot,
    JobStatus,
)
from rag_system.metrics import OperationalMetrics
from rag_system.submission import UploadBatchPreparer
from rag_system.tenancy import Principal


class KnowledgeBaseLifecycle:
    """Own durable creation, deletion, job binding, and restart recovery."""

    def __init__(
        self,
        *,
        settings: Settings,
        service: KnowledgeService,
        catalog: KnowledgeBaseRepository,
        file_store: DocumentStore,
        jobs: JobExecutor,
        idempotency: IdempotencyRepository,
        metrics: OperationalMetrics,
        document_id_factory: Callable[[], str],
    ) -> None:
        self._service = service
        self._catalog = catalog
        self._file_store = file_store
        self._jobs = jobs
        self._idempotency = idempotency
        self._assets = KnowledgeBaseAssets(file_store)
        self._indexing = KnowledgeBaseIndexer(
            service=service,
            catalog=catalog,
            assets=self._assets,
            metrics=metrics,
        )
        self._uploads = UploadBatchPreparer(
            max_file_bytes=settings.max_file_bytes,
            max_total_bytes=settings.max_total_bytes,
            max_documents=settings.max_documents,
            document_id_factory=document_id_factory,
        )
        self._resource_locks = ResourceLockPool(slots=64)
        self._jobs_by_resource = ResourceJobRegistry()

    @property
    def assets(self) -> KnowledgeBaseAssets:
        return self._assets

    @property
    def resource_locks(self) -> ResourceLockPool:
        return self._resource_locks

    def create(
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
        reservation = self._idempotency.reserve(
            principal,
            "knowledge_base.create",
            idempotency_key.strip(),
            request_digest,
        )
        if not reservation.created:
            return self._replay_submission(principal, reservation)

        created_record: KnowledgeBaseRecord | None = None
        job_id: JobId | None = None
        stored_document_ids: frozenset[str] | None = None
        try:
            created_record = self._catalog.create(
                principal,
                display_name,
                idempotency_reservation_id=reservation.reservation_id,
            )
            planned = self._assets.plan(
                principal,
                uploads,
                new_document_id=self._uploads.new_document_id,
            )
            created_record = self._catalog.attach_manifest(
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
            created_record = self._catalog.transition(
                principal,
                created_record.resource_id,
                KnowledgeBaseStatus.PENDING,
            )
            job_id = self._submit_indexing(
                principal,
                created_record.resource_id,
                idempotency_key=reservation.reservation_id,
            )
            self._idempotency.bind_result(
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
                        self.delete(principal, created_record.resource_id)
                        cleanup_complete = True
                    except Exception:
                        pass
            if cleanup_complete:
                try:
                    self._idempotency.abandon(principal, reservation.reservation_id)
                except Exception:
                    pass
            raise
        if created_record is None or job_id is None:
            raise PlatformUnavailableError("knowledge base submission did not complete")
        return KnowledgeBaseSubmission(created_record, job_id, replayed=False)

    def get(self, principal: Principal, resource_id: str) -> KnowledgeBaseRecord:
        return self._catalog.get(principal, resource_id)

    def list(
        self,
        principal: Principal,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[KnowledgeBaseRecord, ...]:
        return self._catalog.list(principal, limit=limit, offset=offset)

    def list_after(
        self,
        principal: Principal,
        *,
        updated_at: float,
        resource_id: str,
        limit: int = 50,
    ) -> tuple[KnowledgeBaseRecord, ...]:
        return self._catalog.list_after(
            principal,
            updated_at=updated_at,
            resource_id=resource_id,
            limit=limit,
        )

    def get_job(self, principal: Principal, job_id: JobId | str) -> JobSnapshot:
        return self._jobs.get(principal.tenant_id.value, job_id)

    def cancel_job(self, principal: Principal, job_id: JobId | str) -> JobSnapshot:
        snapshot = self._jobs.get(principal.tenant_id.value, job_id)
        resource_id = self._resource_for_job(principal, snapshot.job_id)
        if resource_id is not None:
            self._indexing.persist_cancel_intent(principal, resource_id)
        snapshot = self._jobs.cancel(principal.tenant_id.value, snapshot.job_id)
        if resource_id is not None:
            self._indexing.mark_failed(
                principal,
                resource_id,
                KnowledgeBaseErrorCode.INDEX_CANCELLED,
            )
        return snapshot

    def delete(self, principal: Principal, resource_id: str) -> bool:
        job_id = self._job_for_resource(principal, resource_id)
        if job_id is not None:
            try:
                self._jobs.cancel(principal.tenant_id.value, job_id)
            except JobError:
                pass

        with self._resource_locks.hold(resource_id):
            record = self._catalog.get(principal, resource_id)
            if record.status is not KnowledgeBaseStatus.DELETING:
                record = self._catalog.transition(
                    principal,
                    resource_id,
                    KnowledgeBaseStatus.DELETING,
                )
            if record.internal_index_id:
                self._service.index_manager.delete(record.internal_index_id)
            self._assets.delete(principal, record)
            self._catalog.delete(principal, resource_id)
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
            for record in self._recovery_records(principal):
                if record.status is KnowledgeBaseStatus.DELETING:
                    self.delete(principal, record.resource_id)
                    self._abandon_reservation_if_unbound(principal, record)
                    recovered += 1
                elif record.status is KnowledgeBaseStatus.PREPARING:
                    self._recover_preparing(principal, record)
                    recovered += 1
                elif record.status is KnowledgeBaseStatus.CANCELLING:
                    self._catalog.transition(
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

    def _replay_submission(
        self,
        principal: Principal,
        reservation: IdempotencyReservation,
    ) -> KnowledgeBaseSubmission:
        """Converge every durable idempotency replay to one pollable job."""

        record: KnowledgeBaseRecord
        current_job: JobId | None
        needs_binding_recovery = not reservation.is_bound
        if reservation.is_bound:
            bound_resource_id = reservation.resource_id
            bound_job_id = reservation.job_id
            if bound_resource_id is None or bound_job_id is None:
                raise PlatformIntegrityError("bound idempotency result is incomplete")
            record = self._catalog.get(principal, bound_resource_id)
            current_job = self._bound_replay_job(principal, record, bound_job_id)
        else:
            unbound_record = self._catalog.find_by_idempotency_reservation(
                principal,
                reservation.reservation_id,
            )
            if unbound_record is None:
                raise IdempotencyInProgressError("matching request is still in progress")
            record = unbound_record
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
            needs_binding_recovery = True
        if current_job is None:
            raise IdempotencyInProgressError(
                "matching request recovery is still in progress"
            )
        if needs_binding_recovery:
            self._recover_idempotency_binding(principal, record, current_job)
        return KnowledgeBaseSubmission(record, current_job, replayed=True)

    def _bound_replay_job(
        self,
        principal: Principal,
        record: KnowledgeBaseRecord,
        bound_job_id: str,
    ) -> JobId | None:
        current_job = self._job_for_resource(principal, record.resource_id)
        if current_job is not None:
            return current_job
        try:
            snapshot = self._jobs.get(principal.tenant_id.value, JobId(bound_job_id))
        except JobNotFoundError:
            return None
        if (
            record.status is KnowledgeBaseStatus.READY
            and snapshot.status is not JobStatus.SUCCEEDED
        ):
            return None
        return JobId(bound_job_id)

    def _recovery_records(self, principal: Principal) -> Iterator[KnowledgeBaseRecord]:
        records = self._catalog.list(principal, limit=100, offset=0)
        while records:
            yield from records
            if len(records) < 100:
                return
            anchor = records[-1]
            records = self._catalog.list_after(
                principal,
                updated_at=anchor.updated_at,
                resource_id=anchor.resource_id,
                limit=100,
            )

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

        job_id = self._jobs.submit(
            principal.tenant_id.value,
            task,
            idempotency_key=f"recovered-state:{record.resource_id}:{record.version}",
        )
        self._jobs_by_resource.bind(principal.tenant_id.value, record.resource_id, job_id)
        return job_id

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

        job_id = self._jobs.submit(
            principal.tenant_id.value,
            task,
            idempotency_key=idempotency_key,
        )
        self._jobs_by_resource.bind(principal.tenant_id.value, resource_id, job_id)
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
            self._idempotency.recover_binding(
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

    def _rollback_create(
        self,
        principal: Principal,
        record: KnowledgeBaseRecord,
        *,
        stored_document_ids: frozenset[str] | None = None,
    ) -> bool:
        try:
            current = self._catalog.get(principal, record.resource_id)
            if current.status is not KnowledgeBaseStatus.DELETING:
                current = self._catalog.transition(
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
                self._file_store.delete(principal.tenant_id.value, document_id)
            except FileStoreError:
                cleanup_succeeded = False
        if cleanup_succeeded:
            try:
                self._catalog.delete(principal, record.resource_id)
            except Exception:
                return False
            return True
        return False

    def _recover_preparing(
        self,
        principal: Principal,
        record: KnowledgeBaseRecord,
    ) -> None:
        try:
            self._assets.resolve(principal, record)
        except (FileStoreError, PlatformIntegrityError):
            if self._rollback_create(principal, record):
                self._abandon_reservation_if_unbound(principal, record)
            return

        pending = self._catalog.transition(
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
            self._idempotency.abandon(principal, reservation_id)
        except IdempotencyConflictError:
            return

    @staticmethod
    def _namespace(principal: Principal, resource_id: str) -> str:
        return f"{principal.tenant_id.value}:{resource_id}"

    def _job_for_resource(self, principal: Principal, resource_id: str) -> JobId | None:
        tenant_id = principal.tenant_id.value
        job_id = self._jobs_by_resource.job_for(tenant_id, resource_id)
        if job_id is None:
            return None
        try:
            self._jobs.get(tenant_id, job_id)
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
            return self._jobs.get(principal.tenant_id.value, job_id).status.terminal
        except JobNotFoundError:
            return True

    def _resource_for_job(self, principal: Principal, job_id: JobId) -> str | None:
        return self._jobs_by_resource.resource_for(principal.tenant_id.value, job_id)


__all__ = ["KnowledgeBaseLifecycle"]
