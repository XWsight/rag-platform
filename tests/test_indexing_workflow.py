from __future__ import annotations

import threading
import unittest
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from rag_system.application import PlatformIntegrityError, PlatformUnavailableError
from rag_system.domain import IndexRef
from rag_system.indexing import KnowledgeBaseIndexer
from rag_system.ingestion import IngestionResult
from rag_system.job_contracts import CancellationToken, JobCancelledError
from rag_system.knowledge_base_contracts import (
    DocumentManifest,
    KnowledgeBaseErrorCode,
    KnowledgeBaseRecord,
    KnowledgeBaseStatus,
)
from rag_system.metrics import create_operational_metrics
from rag_system.provider_errors import ProviderUnavailableError
from rag_system.security import DocumentValidationError
from rag_system.tenancy import Principal, TenantId


class _Catalog:
    def __init__(self, record: KnowledgeBaseRecord) -> None:
        self.record = record
        self.transitions: list[KnowledgeBaseStatus] = []
        self.fail_transition = False

    def get(self, _principal: Principal, resource_id: str) -> KnowledgeBaseRecord:
        if resource_id != self.record.resource_id:
            raise KeyError(resource_id)
        return self.record

    def transition(
        self,
        _principal: Principal,
        resource_id: str,
        target: KnowledgeBaseStatus,
        *,
        internal_index_id: str | None = None,
        chunk_count: int | None = None,
        error_code: KnowledgeBaseErrorCode | None = None,
    ) -> KnowledgeBaseRecord:
        if resource_id != self.record.resource_id or self.fail_transition:
            raise RuntimeError("catalog unavailable")
        self.transitions.append(target)
        current = self.record
        self.record = replace(
            current,
            status=target,
            internal_index_id=(
                internal_index_id
                if internal_index_id is not None
                else current.internal_index_id
            ),
            chunk_count=chunk_count if chunk_count is not None else current.chunk_count,
            error_code=error_code,
            version=current.version + 1,
            updated_at=current.updated_at + 1,
        )
        return self.record


class _Assets:
    def __init__(self) -> None:
        self.paths = (Path("guide.txt"),)
        self.resolve_error: Exception | None = None

    def resolve(
        self,
        _principal: Principal,
        _record: KnowledgeBaseRecord,
    ) -> tuple[Path, ...]:
        if self.resolve_error is not None:
            raise self.resolve_error
        return self.paths


class _IndexManager:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.delete_error: Exception | None = None

    def delete(self, index_id: str) -> bool:
        self.deleted.append(index_id)
        if self.delete_error is not None:
            raise self.delete_error
        return True


class _Service:
    def __init__(self) -> None:
        self.index_manager = _IndexManager()
        self.prepared = IngestionResult("idx_prepared", (), ())
        self.create_error: Exception | None = None
        self.on_prepare: Callable[[], None] | None = None

    def prepare_index(
        self,
        _paths: list[str],
        *,
        namespace: str,
    ) -> IngestionResult:
        if namespace != "tenant-a:kb_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa":
            raise AssertionError("namespace must stay tenant scoped")
        if self.on_prepare is not None:
            self.on_prepare()
        return self.prepared

    def create_prepared_index(self, ingestion: IngestionResult) -> IndexRef:
        if self.create_error is not None:
            raise self.create_error
        return IndexRef(ingestion.index_id, document_count=1, chunk_count=3, created_at=1.0)


def _principal() -> Principal:
    return Principal(
        subject="operator",
        tenant_id=TenantId("tenant-a"),
        roles=frozenset({"writer"}),
    )


def _record(status: KnowledgeBaseStatus = KnowledgeBaseStatus.PENDING) -> KnowledgeBaseRecord:
    manifest = DocumentManifest("guide.txt", "guide.txt", 7, "a" * 64)
    index_id = "idx_prepared" if status is KnowledgeBaseStatus.INDEXING else None
    return KnowledgeBaseRecord(
        resource_id="kb_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        tenant_id=TenantId("tenant-a"),
        display_name="Guide",
        status=status,
        internal_index_id=index_id,
        documents=(manifest,),
        document_count=1,
        total_bytes=7,
        chunk_count=0,
        error_code=None,
        created_at=1.0,
        updated_at=1.0,
        version=1,
    )


class KnowledgeBaseIndexerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.principal = _principal()
        self.catalog = _Catalog(_record())
        self.assets = _Assets()
        self.service = _Service()
        self.metrics = create_operational_metrics()
        self.indexer = KnowledgeBaseIndexer(
            service=self.service,
            catalog=self.catalog,
            assets=self.assets,
            metrics=self.metrics,
        )

    def test_pending_index_transitions_to_ready_and_records_success(self) -> None:
        result = self.indexer.run(
            self.principal,
            self.catalog.record.resource_id,
            CancellationToken(threading.Event()),
            namespace="tenant-a:kb_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(self.catalog.transitions, [KnowledgeBaseStatus.INDEXING, KnowledgeBaseStatus.READY])
        self.assertEqual(self.catalog.record.status, KnowledgeBaseStatus.READY)
        self.assertEqual(self.catalog.record.chunk_count, 3)
        self.assertEqual(self.service.index_manager.deleted, [])

    def test_provider_failure_marks_failed_cleans_index_and_records_safe_metric(self) -> None:
        self.service.create_error = ProviderUnavailableError("upstream detail")

        with self.assertRaises(ProviderUnavailableError):
            self.indexer.run(
                self.principal,
                self.catalog.record.resource_id,
                CancellationToken(threading.Event()),
                namespace="tenant-a:kb_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )

        self.assertEqual(self.catalog.record.status, KnowledgeBaseStatus.FAILED)
        self.assertEqual(self.catalog.record.error_code, KnowledgeBaseErrorCode.INDEX_BUILD_FAILED)
        self.assertEqual(self.service.index_manager.deleted, ["idx_prepared"])
        metric = self.metrics.external_call_errors_total.snapshot()
        self.assertEqual(metric["series"][0]["labels"]["error_type"], "unavailable")

    def test_cancellation_after_preparation_converges_to_cancelled_failure(self) -> None:
        event = threading.Event()
        self.service.on_prepare = event.set

        with self.assertRaises(JobCancelledError):
            self.indexer.run(
                self.principal,
                self.catalog.record.resource_id,
                CancellationToken(event),
                namespace="tenant-a:kb_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )

        self.assertEqual(self.catalog.record.status, KnowledgeBaseStatus.FAILED)
        self.assertEqual(self.catalog.record.error_code, KnowledgeBaseErrorCode.INDEX_CANCELLED)
        self.assertEqual(self.service.index_manager.deleted, [])

    def test_cancellation_recovery_accepts_an_already_failed_concurrent_transition(self) -> None:
        cancelling = _record(KnowledgeBaseStatus.CANCELLING)
        self.catalog = _Catalog(cancelling)
        self.catalog.fail_transition = True
        self.catalog.record = replace(
            cancelling,
            status=KnowledgeBaseStatus.FAILED,
            error_code=KnowledgeBaseErrorCode.INDEX_CANCELLED,
            updated_at=2.0,
            version=2,
        )
        self.indexer = KnowledgeBaseIndexer(
            service=self.service,
            catalog=self.catalog,
            assets=self.assets,
            metrics=self.metrics,
        )

        recovered = self.indexer.converge_cancel_intent(self.principal, cancelling)

        self.assertEqual(recovered, self.catalog.record)

    def test_recovery_rejects_a_changed_index_identity_and_marks_the_resource_failed(self) -> None:
        self.catalog = _Catalog(_record(KnowledgeBaseStatus.INDEXING))
        self.service.prepared = IngestionResult("idx_different", (), ())
        self.indexer = KnowledgeBaseIndexer(
            service=self.service,
            catalog=self.catalog,
            assets=self.assets,
            metrics=self.metrics,
        )

        with self.assertRaisesRegex(PlatformIntegrityError, "index identity changed"):
            self.indexer.run(
                self.principal,
                self.catalog.record.resource_id,
                CancellationToken(threading.Event()),
                namespace="tenant-a:kb_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )

        self.assertEqual(self.catalog.record.status, KnowledgeBaseStatus.FAILED)
        self.assertEqual(self.catalog.record.error_code, KnowledgeBaseErrorCode.INDEX_BUILD_FAILED)

    def test_document_validation_failure_uses_a_distinct_safe_error_code(self) -> None:
        self.assets.resolve_error = DocumentValidationError("unsafe document")

        with self.assertRaises(DocumentValidationError):
            self.indexer.run(
                self.principal,
                self.catalog.record.resource_id,
                CancellationToken(threading.Event()),
                namespace="tenant-a:kb_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )

        self.assertEqual(self.catalog.record.status, KnowledgeBaseStatus.FAILED)
        self.assertEqual(self.catalog.record.error_code, KnowledgeBaseErrorCode.CONTENT_REJECTED)

    def test_failed_index_cleanup_does_not_hide_the_provider_failure(self) -> None:
        self.service.create_error = ProviderUnavailableError("upstream detail")
        self.service.index_manager.delete_error = RuntimeError("delete failed")

        with self.assertRaises(ProviderUnavailableError):
            self.indexer.run(
                self.principal,
                self.catalog.record.resource_id,
                CancellationToken(threading.Event()),
                namespace="tenant-a:kb_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )

        self.assertEqual(self.catalog.record.status, KnowledgeBaseStatus.FAILED)
        self.assertEqual(self.service.index_manager.deleted, ["idx_prepared"])

    def test_cancel_intent_requires_a_durable_transition(self) -> None:
        self.indexer.persist_cancel_intent(
            self.principal,
            self.catalog.record.resource_id,
        )
        self.assertEqual(self.catalog.record.status, KnowledgeBaseStatus.CANCELLING)

        self.catalog = _Catalog(_record())
        self.catalog.fail_transition = True
        self.indexer = KnowledgeBaseIndexer(
            service=self.service,
            catalog=self.catalog,
            assets=self.assets,
            metrics=self.metrics,
        )
        with self.assertRaisesRegex(PlatformUnavailableError, "cancellation could not be persisted"):
            self.indexer.persist_cancel_intent(
                self.principal,
                self.catalog.record.resource_id,
            )


if __name__ == "__main__":
    unittest.main()
