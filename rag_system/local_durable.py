"""Concrete assembly for the built-in single-node durable runtime profile."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from rag_system.application_ports import (
    DocumentStore,
    IdempotencyRepository,
    JobExecutor,
    KnowledgeBaseRepository,
    KnowledgeService,
)
from rag_system.catalog import KnowledgeBaseCatalog
from rag_system.config import Settings
from rag_system.file_store import TenantFileStore
from rag_system.health import HealthProbe
from rag_system.idempotency import IdempotencyStore
from rag_system.index_manager import IndexManager
from rag_system.job_store import SqliteJobSnapshotStore
from rag_system.jobs import JobManager
from rag_system.ports import IndexRepository
from rag_system.provider_factory import ProviderFactory, create_provider_bundle
from rag_system.providers import ZhipuProviderFactory
from rag_system.rag_service import RagService
from rag_system.retrieval import LocalVectorIndexRepository
from rag_system.runtime_profile import RuntimeComponents
from rag_system.tenancy import Principal


def build_local_service(
    settings: Settings,
    *,
    provider_factory: ProviderFactory | None = None,
    index_repository: IndexRepository | None = None,
) -> RagService:
    """Assemble the built-in service without exposing adapter choices upstream."""

    validated = settings.validate()
    repository = index_repository or LocalVectorIndexRepository(validated)
    manager = IndexManager(validated, repository)
    factory = provider_factory or ZhipuProviderFactory()
    providers = create_provider_bundle(factory, validated)
    return RagService(
        settings=validated,
        index_manager=manager,
        chat_model=providers.chat_model,
        web_search=providers.web_search,
        query_planner=providers.query_planner,
    )


def build_local_durable_components(
    settings: Settings,
    *,
    provider_factory: ProviderFactory | None,
    service_builder: Callable[..., KnowledgeService],
    snapshot_store_factory: type[SqliteJobSnapshotStore] = SqliteJobSnapshotStore,
) -> RuntimeComponents:
    """Build the SQLite/filesystem/thread-pool component set with safe cleanup."""

    service: KnowledgeService | None = None
    jobs: JobExecutor | None = None
    try:
        service = service_builder(settings, provider_factory=provider_factory)
        storage_root = settings.storage_root.expanduser().resolve()
        catalog: KnowledgeBaseRepository = KnowledgeBaseCatalog(storage_root / "catalog.sqlite3")
        file_store: DocumentStore = TenantFileStore(
            storage_root / "documents",
            max_file_bytes=settings.max_file_bytes,
            max_total_bytes=settings.max_tenant_storage_bytes,
            max_files_per_tenant=settings.max_files_per_tenant,
        )
        job_snapshots = snapshot_store_factory(
            storage_root / "jobs.sqlite3",
            ttl_seconds=settings.job_history_ttl_seconds,
            max_records_per_tenant=settings.job_history_max_per_tenant,
        )
        job_snapshots.recover_interrupted()
        jobs = JobManager(
            max_workers=settings.job_workers,
            max_jobs=settings.max_jobs,
            max_jobs_per_tenant=settings.max_jobs_per_tenant,
            ttl_seconds=settings.job_ttl_seconds,
            snapshot_store=job_snapshots,
        )
        idempotency: IdempotencyRepository = IdempotencyStore(
            storage_root / "idempotency.sqlite3",
            ttl_seconds=24 * 60 * 60,
            max_records_per_tenant=10_000,
        )
        return RuntimeComponents(
            service=service,
            catalog=catalog,
            file_store=file_store,
            jobs=jobs,
            idempotency=idempotency,
        )
    except Exception:
        close_unowned_components(service, jobs)
        raise


def local_durable_readiness_probes(
    components: RuntimeComponents,
    principals: Sequence[Principal],
) -> tuple[HealthProbe, ...]:
    return (
        HealthProbe("catalog", lambda: _catalog_ready(components.catalog, principals[0])),
        HealthProbe("documents", components.file_store.healthcheck),
        HealthProbe("jobs", components.jobs.healthcheck),
        HealthProbe("vector", components.service.index_manager.healthcheck),
    )


def _catalog_ready(repository: KnowledgeBaseRepository, principal: Principal) -> bool:
    repository.list(principal, limit=1, offset=0)
    return True


def close_unowned_components(
    service: KnowledgeService | None,
    jobs: JobExecutor | None,
) -> None:
    """Best-effort cleanup while the local profile is only partially assembled."""

    logger = logging.getLogger("rag_system.local_durable")
    if jobs is not None:
        try:
            jobs.shutdown(wait=True, cancel_pending=True)
        except Exception:
            logger.error("job executor cleanup failed during profile assembly")
    if service is None:
        return
    try:
        service.index_manager.close()
    except Exception:
        logger.error("index lifecycle cleanup failed during profile assembly")
    try:
        service.close()
    except Exception:
        logger.error("service cleanup failed during profile assembly")


__all__ = [
    "build_local_durable_components",
    "build_local_service",
    "local_durable_readiness_probes",
]
