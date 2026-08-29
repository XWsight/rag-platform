"""Versioned HTTP schemas and domain-to-wire projections."""

from __future__ import annotations

import math
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rag_system.application_contracts import (
    Application,
    ApplicationDraft,
    ApplicationKind,
    ApplicationRevision,
    ApplicationStatus,
    AuditEvent,
    Deployment,
    DeploymentEnvironment,
    DeploymentStatus,
    Project,
    ResourceAccessMode,
    ResourceBinding,
    ResourceKind,
    RetrievalProfile,
)
from rag_system.application_evaluation import ApplicationEvaluationReport
from rag_system.domain import AnswerClaim, AnswerResult, Citation, Route
from rag_system.grounding import (
    CITATION_ID_PATTERN,
    MAX_ANSWER_CLAIMS,
    MAX_CITATION_ID_CHARACTERS,
    MAX_CLAIM_CHARACTERS,
    MAX_GROUNDED_ANSWER_CHARACTERS,
)
from rag_system.job_contracts import JobSnapshot, JobStatus
from rag_system.knowledge_base_contracts import KnowledgeBaseRecord, KnowledgeBaseStatus


RESOURCE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
SESSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
CitationId = Annotated[
    str,
    Field(
        min_length=2,
        max_length=MAX_CITATION_ID_CHARACTERS,
        pattern=CITATION_ID_PATTERN,
    ),
]


class StrictModel(BaseModel):  # type: ignore[misc]
    """Forbid unknown input fields and implicit type coercion."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class ErrorDetail(StrictModel):
    code: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=256)
    trace_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)


class ErrorEnvelope(StrictModel):
    error: ErrorDetail


class HealthResponse(StrictModel):
    status: Literal["ok", "ready"]


class DocumentResponse(StrictModel):
    name: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class KnowledgeBaseResponse(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=200)
    status: KnowledgeBaseStatus
    documents: tuple[DocumentResponse, ...]
    document_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    error_code: str | None = Field(default=None, max_length=64)
    created_at: float = Field(ge=0)
    updated_at: float = Field(ge=0)
    version: int = Field(ge=1)


class KnowledgeBaseSubmissionResponse(StrictModel):
    knowledge_base: KnowledgeBaseResponse
    job_id: str = Field(min_length=1, max_length=128)
    replayed: bool


class KnowledgeBaseListResponse(StrictModel):
    items: tuple[KnowledgeBaseResponse, ...]
    count: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
    next_cursor: str | None = Field(default=None, max_length=256)


class DeleteResponse(StrictModel):
    deleted: bool


class JobResponse(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    status: JobStatus
    created_at: float = Field(ge=0)
    updated_at: float = Field(ge=0)
    started_at: float | None = Field(default=None, ge=0)
    finished_at: float | None = Field(default=None, ge=0)
    result: dict[str, Any] | None = None
    error_code: str | None = Field(default=None, max_length=64)


class AnswerPayload(StrictModel):
    knowledge_base_id: str = Field(min_length=1, max_length=128, pattern=RESOURCE_PATTERN)
    question: str = Field(min_length=1, max_length=10_000)
    session_id: str = Field(min_length=1, max_length=128, pattern=SESSION_PATTERN)
    allow_cloud: bool = False
    allow_web: bool = False
    deep_research: bool = False


class RouteResponse(StrictModel):
    route: Route
    confidence: float = Field(ge=0, le=1)


class CitationResponse(StrictModel):
    id: CitationId
    source_name: str = Field(min_length=1, max_length=255)
    excerpt: str = Field(max_length=8_000)
    url: str = Field(default="", max_length=4_096)
    score: float | None = None


class AnswerClaimResponse(StrictModel):
    text: str = Field(min_length=1, max_length=MAX_CLAIM_CHARACTERS)
    citation_ids: tuple[CitationId, ...] = Field(max_length=MAX_ANSWER_CLAIMS)


class AnswerResponse(StrictModel):
    answer: str = Field(max_length=MAX_GROUNDED_ANSWER_CHARACTERS)
    decision: RouteResponse
    claims: tuple[AnswerClaimResponse, ...] = Field(max_length=MAX_ANSWER_CLAIMS)
    citations: tuple[CitationResponse, ...]
    trace_id: str = Field(min_length=1, max_length=128)
    latency_ms: float = Field(ge=0)


class ApplicationAnswerPayload(StrictModel):
    question: str = Field(min_length=1, max_length=10_000)
    session_id: str = Field(min_length=1, max_length=128, pattern=SESSION_PATTERN)


class ApplicationDiagnosticsResponse(StrictModel):
    """Small, redaction-safe operational evidence for one application answer."""

    evidence_count: int | None = Field(default=None, ge=0)
    history_turns: int | None = Field(default=None, ge=0)
    planned_query_count: int | None = Field(default=None, ge=0)
    web_query_count: int | None = Field(default=None, ge=0)
    citation_completeness: float | None = Field(default=None, ge=0, le=1)
    planning_error: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    web_error: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    provider_error: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")


class ApplicationAnswerResponse(AnswerResponse):
    application_id: str = Field(min_length=1, max_length=128)
    revision_id: str = Field(min_length=1, max_length=128)
    diagnostics: ApplicationDiagnosticsResponse


class ProjectCreatePayload(StrictModel):
    display_name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2_000)


class ProjectResponse(StrictModel):
    id: str
    display_name: str
    description: str
    created_at: float
    updated_at: float


class ProjectListResponse(StrictModel):
    items: tuple[ProjectResponse, ...]
    count: int = Field(ge=0)


class ApplicationCreatePayload(StrictModel):
    project_id: str
    display_name: str = Field(min_length=1, max_length=200)
    application_kind: Literal["knowledge_chat"] = "knowledge_chat"


class ApplicationResponse(StrictModel):
    id: str
    project_id: str
    display_name: str
    application_kind: ApplicationKind
    active_revision_id: str | None
    status: ApplicationStatus
    created_at: float
    updated_at: float


class ApplicationListResponse(StrictModel):
    items: tuple[ApplicationResponse, ...]
    count: int = Field(ge=0)


class AnswerPolicyPayload(StrictModel):
    require_citations: bool = True
    allow_cloud: bool = False
    allow_web: bool = False
    allow_research: bool = False


class SessionPolicyPayload(StrictModel):
    enabled: bool = True
    ttl_seconds: int | None = Field(default=None, ge=60, le=2_592_000)


class RevisionCreatePayload(StrictModel):
    knowledge_base_ids: list[str] = Field(min_length=1, max_length=32)
    model_profile_id: str = Field(default="default", min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    retrieval_profile: Literal["default", "focused"] = "default"
    answer_policy: AnswerPolicyPayload = AnswerPolicyPayload()
    session_policy: SessionPolicyPayload = SessionPolicyPayload()
    change_summary: str = Field(min_length=1, max_length=500)


class DraftUpdatePayload(StrictModel):
    expected_version: int = Field(ge=0)
    knowledge_base_ids: list[str] = Field(min_length=1, max_length=32)
    model_profile_id: str = Field(default="default", min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    retrieval_profile: Literal["default", "focused"] = "default"
    answer_policy: AnswerPolicyPayload = AnswerPolicyPayload()
    session_policy: SessionPolicyPayload = SessionPolicyPayload()
    change_summary: str = Field(min_length=1, max_length=500)


class DraftRevisionCreatePayload(StrictModel):
    expected_version: int = Field(ge=1)


class DraftResponse(StrictModel):
    application_id: str
    version: int = Field(ge=0)
    configured: bool
    knowledge_base_ids: tuple[str, ...] = ()
    model_profile_id: str | None = None
    retrieval_profile: RetrievalProfile | None = None
    answer_policy: AnswerPolicyPayload | None = None
    session_policy: SessionPolicyPayload | None = None
    updated_at: float
    updated_by: str
    change_summary: str


class RevisionResponse(StrictModel):
    id: str
    application_id: str
    revision_number: int = Field(ge=1)
    configuration_schema_version: int = Field(ge=1)
    knowledge_base_ids: tuple[str, ...]
    model_profile_id: str
    retrieval_profile: RetrievalProfile
    answer_policy: AnswerPolicyPayload
    session_policy: SessionPolicyPayload
    created_at: float
    created_by: str
    change_summary: str


class RevisionListResponse(StrictModel):
    items: tuple[RevisionResponse, ...]
    count: int = Field(ge=0)


class DeploymentCreatePayload(StrictModel):
    revision_id: str
    expected_active_revision_id: str | None


class DeploymentResponse(StrictModel):
    id: str
    application_id: str
    revision_id: str
    environment: DeploymentEnvironment
    status: DeploymentStatus
    deployed_at: float
    deployed_by: str


class DeploymentListResponse(StrictModel):
    items: tuple[DeploymentResponse, ...]
    count: int = Field(ge=0)


class ResourceBindingResponse(StrictModel):
    id: str
    application_id: str
    revision_id: str
    resource_kind: ResourceKind
    resource_id: str
    access_mode: ResourceAccessMode
    created_at: float


class ResourceBindingListResponse(StrictModel):
    items: tuple[ResourceBindingResponse, ...]
    count: int = Field(ge=0)


class AuditEventResponse(StrictModel):
    id: str
    event_type: str
    occurred_at: float
    actor: str
    summary: str
    project_id: str | None
    application_id: str | None
    revision_id: str | None


class AuditEventListResponse(StrictModel):
    items: tuple[AuditEventResponse, ...]
    count: int = Field(ge=0)


class ApplicationEvaluationResponse(StrictModel):
    application_id: str
    revision_id: str
    revision_number: int = Field(ge=1)
    configuration_digest: str = Field(min_length=64, max_length=64)
    generated_at: float = Field(ge=0)
    benchmark: dict[str, Any]


class ApplicationEvaluationCreatePayload(StrictModel):
    """One complete, versioned evaluation report produced by a release pipeline."""

    report: dict[str, Any]


class ApplicationEvaluationListResponse(StrictModel):
    items: tuple[ApplicationEvaluationResponse, ...]
    count: int = Field(ge=0)


def project_response(project: Project) -> ProjectResponse:
    return ProjectResponse(
        id=project.project_id,
        display_name=project.display_name,
        description=project.description,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


def application_response(application: Application) -> ApplicationResponse:
    return ApplicationResponse(
        id=application.application_id,
        project_id=application.project_id,
        display_name=application.display_name,
        application_kind=application.application_kind,
        active_revision_id=application.active_revision_id,
        status=application.status,
        created_at=application.created_at,
        updated_at=application.updated_at,
    )


def revision_response(revision: ApplicationRevision) -> RevisionResponse:
    policy = revision.configuration.answer_policy
    session = revision.configuration.session_policy
    return RevisionResponse(
        id=revision.revision_id,
        application_id=revision.application_id,
        revision_number=revision.revision_number,
        configuration_schema_version=revision.configuration_schema_version,
        knowledge_base_ids=revision.configuration.knowledge_base_ids,
        model_profile_id=revision.configuration.model_profile_id,
        retrieval_profile=revision.configuration.retrieval_profile,
        answer_policy=AnswerPolicyPayload(
            require_citations=policy.require_citations,
            allow_cloud=policy.allow_cloud,
            allow_web=policy.allow_web,
            allow_research=policy.allow_research,
        ),
        session_policy=SessionPolicyPayload(
            enabled=session.enabled, ttl_seconds=session.ttl_seconds
        ),
        created_at=revision.created_at,
        created_by=revision.created_by,
        change_summary=revision.change_summary,
    )


def draft_response(draft: ApplicationDraft) -> DraftResponse:
    configuration = draft.configuration
    if configuration is None:
        return DraftResponse(
            application_id=draft.application_id,
            version=draft.version,
            configured=False,
            updated_at=draft.updated_at,
            updated_by=draft.updated_by,
            change_summary=draft.change_summary,
        )
    policy = configuration.answer_policy
    session = configuration.session_policy
    return DraftResponse(
        application_id=draft.application_id,
        version=draft.version,
        configured=True,
        knowledge_base_ids=configuration.knowledge_base_ids,
        model_profile_id=configuration.model_profile_id,
        retrieval_profile=configuration.retrieval_profile,
        answer_policy=AnswerPolicyPayload(
            require_citations=policy.require_citations,
            allow_cloud=policy.allow_cloud,
            allow_web=policy.allow_web,
            allow_research=policy.allow_research,
        ),
        session_policy=SessionPolicyPayload(enabled=session.enabled, ttl_seconds=session.ttl_seconds),
        updated_at=draft.updated_at,
        updated_by=draft.updated_by,
        change_summary=draft.change_summary,
    )


def deployment_response(deployment: Deployment) -> DeploymentResponse:
    return DeploymentResponse(
        id=deployment.deployment_id,
        application_id=deployment.application_id,
        revision_id=deployment.revision_id,
        environment=deployment.environment,
        status=deployment.status,
        deployed_at=deployment.deployed_at,
        deployed_by=deployment.deployed_by,
    )


def binding_response(binding: ResourceBinding) -> ResourceBindingResponse:
    return ResourceBindingResponse(
        id=binding.binding_id,
        application_id=binding.application_id,
        revision_id=binding.revision_id,
        resource_kind=binding.resource_kind,
        resource_id=binding.resource_id,
        access_mode=binding.access_mode,
        created_at=binding.created_at,
    )


def audit_event_response(event: AuditEvent) -> AuditEventResponse:
    return AuditEventResponse(
        id=event.audit_event_id,
        event_type=event.event_type.value,
        occurred_at=event.occurred_at,
        actor=event.actor,
        summary=event.summary,
        project_id=event.project_id,
        application_id=event.application_id,
        revision_id=event.revision_id,
    )


def application_evaluation_response(
    report: ApplicationEvaluationReport,
) -> ApplicationEvaluationResponse:
    payload = report.to_dict()
    return ApplicationEvaluationResponse(
        application_id=report.application_id,
        revision_id=report.revision_id,
        revision_number=report.revision_number,
        configuration_digest=report.configuration_digest,
        generated_at=report.generated_at,
        benchmark=payload["benchmark"],
    )


def knowledge_base_response(record: KnowledgeBaseRecord) -> KnowledgeBaseResponse:
    documents = tuple(
        DocumentResponse(name=item.display_name, size_bytes=item.size_bytes, sha256=item.sha256)
        for item in record.documents
    )
    return KnowledgeBaseResponse(
        id=record.resource_id,
        name=record.display_name,
        status=record.status,
        documents=documents,
        document_count=record.document_count,
        total_bytes=record.total_bytes,
        chunk_count=record.chunk_count,
        error_code=record.error_code.value if record.error_code is not None else None,
        created_at=record.created_at,
        updated_at=record.updated_at,
        version=record.version,
    )


def job_response(snapshot: JobSnapshot) -> JobResponse:
    return JobResponse(
        id=snapshot.job_id.value,
        status=snapshot.status,
        created_at=snapshot.created_at,
        updated_at=snapshot.updated_at,
        started_at=snapshot.started_at,
        finished_at=snapshot.finished_at,
        result=snapshot.result,
        error_code=snapshot.error_code or None,
    )


def citation_response(citation: Citation) -> CitationResponse:
    return CitationResponse(
        id=citation.citation_id,
        source_name=citation.source_name,
        excerpt=citation.excerpt,
        url=citation.url,
        score=citation.score,
    )


def claim_response(claim: AnswerClaim) -> AnswerClaimResponse:
    return AnswerClaimResponse(text=claim.text, citation_ids=claim.citation_ids)


def answer_response(result: AnswerResult, *, trace_id: str) -> AnswerResponse:
    return AnswerResponse(
        answer=result.answer,
        decision=RouteResponse(
            route=result.decision.route,
            confidence=max(0.0, min(1.0, result.decision.confidence)),
        ),
        claims=tuple(claim_response(item) for item in result.claims),
        citations=tuple(citation_response(item) for item in result.citations),
        trace_id=trace_id,
        latency_ms=max(0.0, result.latency_ms),
    )


def application_diagnostics_response(result: AnswerResult) -> ApplicationDiagnosticsResponse:
    """Expose only known bounded diagnostic values, never opaque provider detail."""

    diagnostics = result.diagnostics

    def nonnegative_int(name: str) -> int | None:
        value = diagnostics.get(name)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    def fraction(name: str) -> float | None:
        value = diagnostics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        normalized = float(value)
        return normalized if math.isfinite(normalized) and 0 <= normalized <= 1 else None

    def error_code(name: str) -> str | None:
        value = diagnostics.get(name)
        return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", value) else None

    return ApplicationDiagnosticsResponse(
        evidence_count=nonnegative_int("evidence_count"),
        history_turns=nonnegative_int("history_turns"),
        planned_query_count=nonnegative_int("planned_query_count"),
        web_query_count=nonnegative_int("web_query_count"),
        citation_completeness=fraction("citation_completeness"),
        planning_error=error_code("planning_error"),
        web_error=error_code("web_error"),
        provider_error=error_code("provider_error"),
    )


__all__ = [
    "AnswerPayload",
    "AnswerResponse",
    "ApplicationAnswerPayload",
    "ApplicationAnswerResponse",
    "ApplicationDiagnosticsResponse",
    "ApplicationEvaluationCreatePayload",
    "ApplicationEvaluationListResponse",
    "ApplicationEvaluationResponse",
    "ApplicationCreatePayload",
    "ApplicationListResponse",
    "ApplicationResponse",
    "AuditEventListResponse",
    "AuditEventResponse",
    "AnswerPolicyPayload",
    "DeploymentCreatePayload",
    "DeploymentListResponse",
    "DeploymentResponse",
    "ProjectCreatePayload",
    "ProjectListResponse",
    "ProjectResponse",
    "RevisionCreatePayload",
    "RevisionListResponse",
    "RevisionResponse",
    "ResourceBindingListResponse",
    "ResourceBindingResponse",
    "SessionPolicyPayload",
    "DeleteResponse",
    "DocumentResponse",
    "DraftResponse",
    "DraftRevisionCreatePayload",
    "DraftUpdatePayload",
    "ErrorDetail",
    "ErrorEnvelope",
    "HealthResponse",
    "JobResponse",
    "KnowledgeBaseListResponse",
    "KnowledgeBaseResponse",
    "KnowledgeBaseSubmissionResponse",
    "RESOURCE_PATTERN",
    "SESSION_PATTERN",
    "answer_response",
    "application_diagnostics_response",
    "application_evaluation_response",
    "application_response",
    "audit_event_response",
    "binding_response",
    "deployment_response",
    "draft_response",
    "project_response",
    "revision_response",
    "job_response",
    "knowledge_base_response",
]
