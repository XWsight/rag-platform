"""Replaceable production-runtime profiles and their owned dependencies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_system.application_ports import (
    DocumentStore,
    IdempotencyRepository,
    JobExecutor,
    KnowledgeBaseRepository,
    KnowledgeService,
)
from rag_system.config import Settings
from rag_system.health import HealthProbe
from rag_system.provider_factory import ProviderFactory
from rag_system.tenancy import Principal


@dataclass(frozen=True, slots=True)
class RuntimeComponents:
    """Resources owned by a production profile until a platform takes them over."""

    service: KnowledgeService
    catalog: KnowledgeBaseRepository
    file_store: DocumentStore
    jobs: JobExecutor
    idempotency: IdempotencyRepository

    def close(self) -> None:
        """Release partially assembled resources without hiding the first failure."""

        failure: Exception | None = None
        closers: tuple[Callable[[], None], ...] = (
            lambda: self.jobs.shutdown(wait=True, cancel_pending=True),
            self.service.index_manager.close,
            self.service.close,
        )
        for close in closers:
            try:
                close()
            except Exception as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure


class RuntimeProfile(Protocol):
    """Assemble one durable deployment shape without changing application flows."""

    def build_components(
        self,
        settings: Settings,
        *,
        provider_factory: ProviderFactory | None = None,
    ) -> RuntimeComponents: ...

    def readiness_probes(
        self,
        components: RuntimeComponents,
        principals: Sequence[Principal],
    ) -> tuple[HealthProbe, ...]: ...


__all__ = ["RuntimeComponents", "RuntimeProfile"]
