"""Tenant-scoped answer orchestration independent of delivery adapters."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import replace

from rag_system.application import (
    KnowledgeBaseNotReadyError,
    PlatformIntegrityError,
    PlatformUnavailableError,
    PlatformValidationError,
)
from rag_system.application_ports import KnowledgeBaseRepository, KnowledgeService
from rag_system.knowledge_base_assets import KnowledgeBaseAssets
from rag_system.config import Settings
from rag_system.coordination import ResourceLockPool
from rag_system.domain import AnswerRequest, AnswerResult
from rag_system.metrics import OperationalMetrics
from rag_system.knowledge_base_contracts import KnowledgeBaseStatus
from rag_system.tenancy import Principal


class KnowledgeBaseAnswerWorkflow:
    """Serve bounded answers while preserving catalog and index invariants."""

    def __init__(
        self,
        *,
        settings: Settings,
        catalog: KnowledgeBaseRepository,
        service: KnowledgeService,
        assets: KnowledgeBaseAssets,
        resource_locks: ResourceLockPool,
        metrics: OperationalMetrics,
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._service = service
        self._assets = assets
        self._resource_locks = resource_locks
        self._metrics = metrics
        self._slots = threading.BoundedSemaphore(settings.max_concurrent_answers)

    def answer(
        self,
        principal: Principal,
        resource_id: str,
        request: AnswerRequest,
    ) -> AnswerResult:
        if not self._slots.acquire(blocking=False):
            raise PlatformUnavailableError("answer capacity is temporarily exhausted")
        try:
            return self._answer_unbounded(principal, resource_id, request)
        finally:
            self._slots.release()

    def clear_session(
        self,
        principal: Principal,
        resource_id: str,
        session_id: str,
    ) -> bool:
        self._catalog.get(principal, resource_id)
        return self._service.clear_session(
            self._session_id(principal, resource_id, session_id)
        )

    def _answer_unbounded(
        self,
        principal: Principal,
        resource_id: str,
        request: AnswerRequest,
    ) -> AnswerResult:
        record = self._catalog.get(principal, resource_id)
        if record.status is not KnowledgeBaseStatus.READY or not record.internal_index_id:
            raise KnowledgeBaseNotReadyError("knowledge base is not ready")

        try:
            self._service.index_manager.get(record.internal_index_id)
        except KeyError:
            with self._resource_locks.hold(resource_id):
                record = self._catalog.get(principal, resource_id)
                if (
                    record.status is not KnowledgeBaseStatus.READY
                    or not record.internal_index_id
                ):
                    raise KnowledgeBaseNotReadyError(
                        "knowledge base is not ready"
                    ) from None
                paths = self._assets.resolve(principal, record)
                restored = self._service.create_index(
                    [str(path) for path in paths],
                    namespace=f"{principal.tenant_id.value}:{resource_id}",
                )
                if restored.index_id != record.internal_index_id:
                    self._service.index_manager.delete(restored.index_id)
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
            result = self._service.answer(internal_index_id, scoped_request)
            self._record_external_call_failures(result)
            return result
        except KeyError:
            current = self._catalog.get(principal, resource_id)
            if current.status is not KnowledgeBaseStatus.READY:
                raise KnowledgeBaseNotReadyError("knowledge base is not ready") from None
            raise KnowledgeBaseNotReadyError(
                "knowledge base index is being reloaded"
            ) from None

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
        remaining = self._settings.research_max_web_queries
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
        bounded_count = (
            count if isinstance(count, int) and not isinstance(count, bool) else 1
        )
        bounded_count = min(
            max(bounded_count, 1),
            self._settings.research_max_web_queries,
        )
        try:
            self._metrics.external_call_errors_total.increment(
                amount=bounded_count,
                labels={
                    "provider": provider,
                    "operation": operation,
                    "error_type": error_type,
                },
            )
        except Exception:
            return

    @staticmethod
    def _session_id(principal: Principal, resource_id: str, session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise PlatformValidationError("session_id is required")
        if len(session_id) > 128:
            raise PlatformValidationError("session_id is too long")
        identity = f"{principal.tenant_id.value}\0{resource_id}\0{session_id.strip()}"
        return "session_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


__all__ = ["KnowledgeBaseAnswerWorkflow"]
