"""Runtime resolution for deployed, tenant-scoped AI applications."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from rag_system.application import RagApplication
from rag_system.application_contracts import ApplicationStatus, KnowledgeChatConfiguration
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


class KnowledgeApplicationRuntime:
    """Resolve a deployed knowledge-chat application through the existing RAG facade."""

    def __init__(self, applications: ApplicationRepository, rag: RagApplication) -> None:
        self._applications = applications
        self._rag = rag

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
        resource_id = self._resolve_bound_knowledge_base(
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
        )
        try:
            result = self._rag.answer(principal, resource_id, request)
        finally:
            if not configuration.session_policy.enabled:
                try:
                    self._rag.clear_session(principal, resource_id, runtime_session_id)
                except Exception:
                    pass
        return ApplicationAnswer(
            application_id=application.application_id,
            revision_id=revision.revision_id,
            result=result,
        )

    def _resolve_bound_knowledge_base(
        self,
        principal: Principal,
        application_id: str,
        revision_id: str,
        configuration: KnowledgeChatConfiguration,
    ) -> str:
        bindings = self._applications.list_bindings(principal, application_id, revision_id)
        bound_ids = tuple(binding.resource_id for binding in bindings)
        if bound_ids != configuration.knowledge_base_ids or len(bound_ids) != 1:
            raise ApplicationBoundResourceUnavailableError()
        resource_id = bound_ids[0]
        try:
            record = self._rag.get_knowledge_base(principal, resource_id)
        except Exception as error:
            raise ApplicationBoundResourceUnavailableError() from error
        if record.status is not KnowledgeBaseStatus.READY:
            raise ApplicationBoundResourceUnavailableError()
        return resource_id


def _runtime_session_id(application_id: str, revision_id: str, session_id: object) -> str:
    if not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 128:
        raise ApplicationRuntimeValidationError("session_id has an invalid length.")
    material = f"{application_id}\0{revision_id}\0{session_id.strip()}".encode()
    return "appsession_" + hashlib.sha256(material).hexdigest()


__all__ = [
    "ApplicationAnswer",
    "ApplicationBoundResourceUnavailableError",
    "ApplicationNotPublishedError",
    "ApplicationRuntimeError",
    "ApplicationRuntimeValidationError",
    "KnowledgeApplicationRuntime",
]
