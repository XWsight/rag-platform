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
from rag_system.health import HealthProbe, ReadinessMonitor
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


@dataclass(frozen=True, slots=True)
class RuntimeProfileVerification:
    """Non-sensitive result from an isolated profile preflight."""

    probe_names: tuple[str, ...]


class RuntimeProfileConformanceError(ValueError):
    """Raised when a custom runtime profile cannot satisfy the base contract."""


def verify_runtime_profile(
    profile: RuntimeProfile,
    settings: Settings,
    principals: Sequence[Principal],
    *,
    provider_factory: ProviderFactory | None = None,
) -> RuntimeProfileVerification:
    """Build, probe, and release a profile in an isolated disposable environment.

    Callers should supply a disposable storage root in ``settings``.  The helper
    intentionally does not start recovery workflows or accept production
    traffic; it verifies only the profile-owned assembly, readiness, and
    cleanup contract.
    """

    try:
        components = profile.build_components(settings, provider_factory=provider_factory)
    except Exception as error:
        raise RuntimeProfileConformanceError("runtime profile component assembly failed") from error
    if not isinstance(components, RuntimeComponents):
        raise RuntimeProfileConformanceError("runtime profile must return RuntimeComponents")

    failure: Exception | None = None
    verification: RuntimeProfileVerification | None = None
    try:
        probes = profile.readiness_probes(components, principals)
        snapshot = ReadinessMonitor(probes).snapshot()
        unavailable = tuple(item.name for item in snapshot if not item.ready)
        if unavailable:
            raise RuntimeProfileConformanceError(
                "runtime profile readiness probes are unavailable: " + ", ".join(unavailable)
            )
        verification = RuntimeProfileVerification(tuple(item.name for item in snapshot))
    except Exception as error:
        failure = error
    try:
        components.close()
    except Exception as error:
        if failure is None:
            raise RuntimeProfileConformanceError("runtime profile component cleanup failed") from error
    if failure is not None:
        if isinstance(failure, RuntimeProfileConformanceError):
            raise failure
        raise RuntimeProfileConformanceError("runtime profile readiness validation failed") from failure
    assert verification is not None
    return verification


__all__ = [
    "RuntimeComponents",
    "RuntimeProfile",
    "RuntimeProfileConformanceError",
    "RuntimeProfileVerification",
    "verify_runtime_profile",
]
