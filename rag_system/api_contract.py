"""Versioned HTTP schemas and domain-to-wire projections."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rag_system.application_contracts import (
    Application,
    ApplicationDraft,
    ApplicationKind,
    ApplicationRevision,
    ApplicationStatus,
    Deployment,
    DeploymentEnvironment,
    Project,
    RetrievalProfile,
)
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


class ApplicationAnswerResponse(AnswerResponse):
    application_id: str = Field(min_length=1, max_length=128)
    revision_id: str = Field(min_length=1, max_length=128)


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
    retrieval_profile: Literal["default", "focused"] = "default"
    answer_policy: AnswerPolicyPayload = AnswerPolicyPayload()
    session_policy: SessionPolicyPayload = SessionPolicyPayload()
    change_summary: str = Field(min_length=1, max_length=500)


class DraftUpdatePayload(StrictModel):
    expected_version: int = Field(ge=0)
    knowledge_base_ids: list[str] = Field(min_length=1, max_length=32)
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
    deployed_at: float
    deployed_by: str


class DeploymentListResponse(StrictModel):
    items: tuple[DeploymentResponse, ...]
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
        deployed_at=deployment.deployed_at,
        deployed_by=deployment.deployed_by,
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


__all__ = [
    "AnswerPayload",
    "AnswerResponse",
    "ApplicationAnswerPayload",
    "ApplicationAnswerResponse",
    "ApplicationCreatePayload",
    "ApplicationListResponse",
    "ApplicationResponse",
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
    "application_response",
    "deployment_response",
    "draft_response",
    "project_response",
    "revision_response",
    "job_response",
    "knowledge_base_response",
]
