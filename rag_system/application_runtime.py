"""Runtime resolution for deployed, tenant-scoped AI applications."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Protocol

from rag_system.application import RagApplication
from rag_system.application_contracts import (
    ApplicationStatus,
    KnowledgeChatConfiguration,
    validate_model_profile_id,
)
from rag_system.application_ports import ApplicationRepository
from rag_system.domain import AnswerRequest, AnswerResult
from rag_system.knowledge_base_contracts import KnowledgeBaseStatus
from rag_system.tenancy import Principal


class ApplicationRuntimeError(Exception):
    """Base error for a safe application-runtime refusal."""


class ApplicationNotPublishedError(ApplicationRuntimeError):
    def __init__(self) -> None:
        super().__init__("Application is not published.")


class ApplicationBoundResourceUnavailableError(ApplicationRuntimeError):
    def __init__(self) -> None:
        super().__init__("A bound application resource is unavailable.")


class ApplicationRuntimeValidationError(ApplicationRuntimeError, ValueError):
    """The caller's runtime request does not satisfy the application contract."""


@dataclass(frozen=True, slots=True)
class ApplicationAnswer:
    """An answer paired with the immutable application revision that produced it."""

    application_id: str
    revision_id: str
    result: AnswerResult


class ApplicationRuntime(Protocol):
    """Stable runtime port for application-kind adapters."""

    def answer(
        self,
        principal: Principal,
        application_id: str,
        *,
        question: str,
        session_id: str,
    ) -> ApplicationAnswer: ...


class KnowledgeApplicationRuntime:
    """Resolve a deployed knowledge-chat application through the existing RAG facade."""

    def __init__(
        self,
        applications: ApplicationRepository,
        rag: RagApplication,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_tracked_sessions: int = 4_096,
        trusted_model_profile_ids: Sequence[str] = ("default",),
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        if isinstance(max_tracked_sessions, bool) or max_tracked_sessions < 1:
            raise ValueError("max_tracked_sessions must be positive")
        self._applications = applications
        self._rag = rag
        self._clock = clock
        self._max_tracked_sessions = max_tracked_sessions
        profile_ids = tuple(validate_model_profile_id(value) for value in trusted_model_profile_ids)
        if not profile_ids or len(set(profile_ids)) != len(profile_ids):
            raise ValueError("trusted_model_profile_ids must contain unique profile IDs")
        self._trusted_model_profile_ids = frozenset(profile_ids)
        self._session_accessed_at: OrderedDict[str, float] = OrderedDict()
        self._session_lock = RLock()

    def answer(
        self,
        principal: Principal,
        application_id: str,
        *,
        question: str,
        session_id: str,
    ) -> ApplicationAnswer:
        application = self._applications.get_application(principal, application_id)
        if application.status is not ApplicationStatus.ACTIVE or application.active_revision_id is None:
            raise ApplicationNotPublishedError()
        revision = self._applications.get_revision(
            principal, application.application_id, application.active_revision_id
        )
        configuration = revision.configuration
        if configuration.model_profile_id not in self._trusted_model_profile_ids:
            raise ApplicationRuntimeValidationError("The model profile is unavailable.")
        resource_ids = self._resolve_bound_knowledge_bases(
            principal, application.application_id, revision.revision_id, configuration
        )
        runtime_session_id = _runtime_session_id(
            application.application_id, revision.revision_id, session_id
        )
        request = AnswerRequest(
            question=question,
            session_id=runtime_session_id,
            allow_cloud=configuration.answer_policy.allow_cloud,
            allow_web=configuration.answer_policy.allow_web,
            deep_research=configuration.answer_policy.allow_research,
            require_citations=configuration.answer_policy.require_citations,
            retrieval_profile=configuration.retrieval_profile.value,
        )
        if configuration.session_policy.enabled:
            self._clear_expired_session(
                principal,
                resource_ids,
                runtime_session_id,
                configuration.session_policy.ttl_seconds,
            )
        try:
            result = (
                self._rag.answer(principal, resource_ids[0], request)
                if len(resource_ids) == 1
                else self._rag.answer_across_knowledge_bases(principal, resource_ids, request)
            )
        finally:
            if not configuration.session_policy.enabled:
                self._clear_bound_session(principal, resource_ids, runtime_session_id)
            else:
                self._touch_session(runtime_session_id)
        return ApplicationAnswer(
            application_id=application.application_id,
            revision_id=revision.revision_id,
            result=result,
        )

    def _resolve_bound_knowledge_bases(
        self,
        principal: Principal,
        application_id: str,
        revision_id: str,
        configuration: KnowledgeChatConfiguration,
    ) -> tuple[str, ...]:
        bindings = self._applications.list_bindings(principal, application_id, revision_id)
        bound_ids = tuple(binding.resource_id for binding in bindings)
        if len(bound_ids) != len(configuration.knowledge_base_ids) or set(bound_ids) != set(
            configuration.knowledge_base_ids
        ):
            raise ApplicationBoundResourceUnavailableError()
        for resource_id in configuration.knowledge_base_ids:
            try:
                record = self._rag.get_knowledge_base(principal, resource_id)
            except Exception as error:
                raise ApplicationBoundResourceUnavailableError() from error
            if record.status is not KnowledgeBaseStatus.READY:
                raise ApplicationBoundResourceUnavailableError()
        return configuration.knowledge_base_ids

    def _clear_expired_session(
        self,
        principal: Principal,
        resource_ids: tuple[str, ...],
        session_id: str,
        ttl_seconds: int | None,
    ) -> None:
        if ttl_seconds is None:
            return
        now = float(self._clock())
        with self._session_lock:
            last_accessed_at = self._session_accessed_at.pop(session_id, None)
        if last_accessed_at is not None and now - last_accessed_at >= ttl_seconds:
            self._clear_bound_session(principal, resource_ids, session_id)

    def _touch_session(self, session_id: str) -> None:
        now = float(self._clock())
        with self._session_lock:
            self._session_accessed_at[session_id] = now
            self._session_accessed_at.move_to_end(session_id)
            while len(self._session_accessed_at) > self._max_tracked_sessions:
                self._session_accessed_at.popitem(last=False)

    def _clear_bound_session(
        self, principal: Principal, resource_ids: tuple[str, ...], session_id: str
    ) -> None:
        try:
            if len(resource_ids) == 1:
                self._rag.clear_session(principal, resource_ids[0], session_id)
            else:
                self._rag.clear_session_across_knowledge_bases(
                    principal, resource_ids, session_id
                )
        except Exception:
            return


def _runtime_session_id(application_id: str, revision_id: str, session_id: object) -> str:
    if not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 128:
        raise ApplicationRuntimeValidationError("session_id has an invalid length.")
    material = f"{application_id}\0{revision_id}\0{session_id.strip()}".encode()
    return "appsession_" + hashlib.sha256(material).hexdigest()


__all__ = [
    "ApplicationAnswer",
    "ApplicationBoundResourceUnavailableError",
    "ApplicationNotPublishedError",
    "ApplicationRuntime",
    "ApplicationRuntimeError",
    "ApplicationRuntimeValidationError",
    "KnowledgeApplicationRuntime",
]
