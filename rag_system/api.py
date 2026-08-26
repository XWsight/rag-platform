"""Authenticated, tenant-scoped HTTP boundary for the production service."""

import base64
import json
import math
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, Path, Query, Request, UploadFile
from fastapi.responses import Response
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from rag_system.api_contract import (
    RESOURCE_PATTERN as _RESOURCE_PATTERN,
    SESSION_PATTERN as _SESSION_PATTERN,
    AnswerPayload,
    AnswerResponse,
    DeleteResponse,
    DocumentResponse,
    ErrorDetail,
    ErrorEnvelope,
    HealthResponse,
    JobResponse,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
    KnowledgeBaseSubmissionResponse,
    answer_response,
    job_response,
    knowledge_base_response,
)
from rag_system.api_error_handlers import install_error_handlers
from rag_system.api_errors import ApiBoundaryError
from rag_system.api_request_context import install_request_context_middleware, outcome_for as _outcome_for
from rag_system.api_security import build_api_security_dependencies
from rag_system.application import RagApplication
from rag_system.api_uploads import read_uploads as _read_uploads
from rag_system.domain import AnswerRequest
from rag_system.api_responses import (
    context_from_request as _context_from_request,
)
from rag_system.observability import JsonEventLogger
from rag_system.rate_limit import TokenBucketRateLimiter
from rag_system.tenancy import (
    ApiKeyAuthenticator,
    Principal,
)
from rag_system.api_openapi import install_multipart_openapi_schema
from rag_system.web_ui import mount_web_ui


_PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


ReadinessCheck = Callable[[], bool]


def _error_responses(*status_codes: int) -> dict[int, dict[str, object]]:
    """Declare the service error envelope consistently in OpenAPI."""

    return {status_code: {"model": ErrorEnvelope} for status_code in status_codes}


def create_app(
    *,
    platform: RagApplication,
    authenticator: ApiKeyAuthenticator,
    rate_limiter: TokenBucketRateLimiter,
    logger: JsonEventLogger,
    readiness: ReadinessCheck | bool | None = None,
    shutdown: Callable[[], None] | None = None,
    close_on_shutdown: bool = True,
) -> FastAPI:
    """Build an isolated FastAPI application from explicit dependencies."""

    if platform is None or authenticator is None or rate_limiter is None or logger is None:
        raise TypeError("platform, authenticator, rate_limiter, and logger are required")
    if readiness is not None and not isinstance(readiness, bool) and not callable(readiness):
        raise TypeError("readiness must be a boolean or callable")
    if shutdown is not None and not callable(shutdown):
        raise TypeError("shutdown must be callable")

    settings = platform.settings.validate()
    docs_enabled = settings.api_docs_enabled

    class ConfiguredAnswerPayload(AnswerPayload):
        question: str = Field(
            min_length=1,
            max_length=settings.max_question_characters,
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if close_on_shutdown:
            await run_in_threadpool(shutdown or platform.close)

    app = FastAPI(
        title=f"{settings.product_name} API",
        summary="Tenant-isolated, grounded knowledge retrieval and answering.",
        version="2.0.0",
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
        lifespan=lifespan,
    )
    mount_web_ui(
        app,
        product_name=settings.product_name,
        product_tagline=settings.product_tagline,
    )

    install_multipart_openapi_schema(app)

    security = build_api_security_dependencies(
        authenticator=authenticator,
        rate_limiter=rate_limiter,
    )

    install_request_context_middleware(
        app,
        platform=platform,
        settings=settings,
        security=security,
        logger=logger,
    )

    install_error_handlers(app, logger=logger, outcome_for=_outcome_for)

    reader = security.reader
    writer = security.writer
    operator = security.operator
    consume = security.consume

    @app.get(
        "/health/live",
        response_model=HealthResponse,
        tags=["health"],
        summary="Process liveness",
    )
    def live(request: Request) -> HealthResponse:
        request.state.operation = "health"
        return HealthResponse(status="ok")

    @app.get(
        "/health/ready",
        response_model=HealthResponse,
        responses=_error_responses(500, 503),
        tags=["health"],
        summary="Local storage and job-manager readiness",
    )
    def ready(request: Request) -> HealthResponse:
        request.state.operation = "health"
        try:
            available = True if readiness is None else (
                readiness if isinstance(readiness, bool) else bool(readiness())
            )
        except Exception:
            available = False
        if not available:
            raise ApiBoundaryError(503, "not_ready", "The service is not ready.")
        return HealthResponse(status="ready")

    @app.post(
        "/v1/knowledge-bases",
        status_code=202,
        response_model=KnowledgeBaseSubmissionResponse,
        responses=_error_responses(401, 403, 409, 413, 422, 429, 500, 503),
        tags=["knowledge bases"],
        summary="Create and asynchronously index a knowledge base",
    )
    async def create_knowledge_base(
        request: Request,
        principal: Annotated[Principal, Depends(writer)],
        name: Annotated[str, Form(min_length=1, max_length=200)],
        files: Annotated[list[UploadFile], File(description="Documents to index")],
        idempotency_key: Annotated[
            str,
            Header(
                alias="Idempotency-Key",
                min_length=8,
                max_length=128,
                pattern=r"^[!-~]{8,128}$",
            ),
        ],
    ) -> KnowledgeBaseSubmissionResponse:
        request.state.operation = "ingest"
        consume(request, principal, tokens=2)
        display_name = name.strip()
        if not display_name:
            raise ApiBoundaryError(422, "invalid_request", "The request could not be validated.")
        if not 1 <= len(files) <= settings.max_documents:
            raise ApiBoundaryError(
                413,
                "upload_limit_exceeded",
                "The upload exceeds the configured limits.",
            )
        uploads = await _read_uploads(
            files,
            max_file_bytes=settings.max_file_bytes,
            max_total_bytes=settings.max_total_bytes,
        )
        submission = await run_in_threadpool(
            platform.create_knowledge_base,
            principal,
            display_name=display_name,
            documents=uploads,
            idempotency_key=idempotency_key.strip(),
        )
        return KnowledgeBaseSubmissionResponse(
            knowledge_base=knowledge_base_response(submission.knowledge_base),
            job_id=submission.job_id.value,
            replayed=submission.replayed,
        )

    @app.get(
        "/v1/knowledge-bases",
        response_model=KnowledgeBaseListResponse,
        responses=_error_responses(401, 403, 422, 429, 500, 503),
        tags=["knowledge bases"],
        summary="List knowledge bases for the authenticated tenant",
    )
    def list_knowledge_bases(
        request: Request,
        principal: Annotated[Principal, Depends(reader)],
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0, le=10_000)] = 0,
        cursor: Annotated[str | None, Query(max_length=256)] = None,
    ) -> KnowledgeBaseListResponse:
        request.state.operation = "index"
        consume(request, principal)
        if cursor is None:
            records = platform.list_knowledge_bases(principal, limit=limit, offset=offset)
        else:
            if offset != 0:
                raise ApiBoundaryError(422, "invalid_request", "Cursor and offset cannot be combined.")
            updated_at, resource_id = _decode_knowledge_base_cursor(cursor)
            records = platform.list_knowledge_bases_after(
                principal,
                updated_at=updated_at,
                resource_id=resource_id,
                limit=limit,
            )
        items = tuple(knowledge_base_response(record) for record in records)
        next_cursor = _encode_knowledge_base_cursor(items[-1]) if len(items) == limit else None
        return KnowledgeBaseListResponse(
            items=items,
            count=len(items),
            limit=limit,
            offset=offset,
            next_cursor=next_cursor,
        )

    @app.get(
        "/v1/knowledge-bases/{knowledge_base_id}",
        response_model=KnowledgeBaseResponse,
        responses=_error_responses(401, 403, 404, 422, 429, 500, 503),
        tags=["knowledge bases"],
        summary="Get a tenant-owned knowledge base",
    )
    def get_knowledge_base(
        request: Request,
        principal: Annotated[Principal, Depends(reader)],
        knowledge_base_id: Annotated[str, Path(pattern=_RESOURCE_PATTERN, max_length=128)],
    ) -> KnowledgeBaseResponse:
        request.state.operation = "index"
        consume(request, principal)
        return knowledge_base_response(platform.get_knowledge_base(principal, knowledge_base_id))

    @app.delete(
        "/v1/knowledge-bases/{knowledge_base_id}",
        response_model=DeleteResponse,
        responses=_error_responses(401, 403, 404, 409, 422, 429, 500, 503),
        tags=["knowledge bases"],
        summary="Delete a knowledge base and its stored data",
    )
    def delete_knowledge_base(
        request: Request,
        principal: Annotated[Principal, Depends(writer)],
        knowledge_base_id: Annotated[str, Path(pattern=_RESOURCE_PATTERN, max_length=128)],
    ) -> DeleteResponse:
        request.state.operation = "index"
        consume(request, principal, tokens=2)
        deleted = platform.delete_knowledge_base(principal, knowledge_base_id)
        return DeleteResponse(deleted=bool(deleted))

    @app.get(
        "/v1/jobs/{job_id}",
        response_model=JobResponse,
        responses=_error_responses(401, 403, 404, 422, 429, 500, 503),
        tags=["jobs"],
        summary="Get background job state",
    )
    def get_job(
        request: Request,
        principal: Annotated[Principal, Depends(reader)],
        job_id: Annotated[str, Path(pattern=_RESOURCE_PATTERN, max_length=128)],
    ) -> JobResponse:
        request.state.operation = "index"
        consume(request, principal)
        return job_response(platform.get_job(principal, job_id))

    @app.delete(
        "/v1/jobs/{job_id}",
        response_model=JobResponse,
        responses=_error_responses(401, 403, 404, 409, 422, 429, 500, 503),
        tags=["jobs"],
        summary="Request cooperative job cancellation",
    )
    def cancel_job(
        request: Request,
        principal: Annotated[Principal, Depends(writer)],
        job_id: Annotated[str, Path(pattern=_RESOURCE_PATTERN, max_length=128)],
    ) -> JobResponse:
        request.state.operation = "index"
        consume(request, principal)
        return job_response(platform.cancel_job(principal, job_id))

    @app.post(
        "/v1/answers",
        response_model=AnswerResponse,
        responses=_error_responses(401, 403, 404, 409, 422, 429, 500, 503),
        tags=["answers"],
        summary="Answer from a tenant-owned knowledge base",
    )
    def answer(
        request: Request,
        principal: Annotated[Principal, Depends(reader)],
        payload: ConfiguredAnswerPayload,
    ) -> AnswerResponse:
        request.state.operation = "research" if payload.deep_research else "answer"
        consume(request, principal, tokens=3 if payload.deep_research else 1)
        result = platform.answer(
            principal,
            payload.knowledge_base_id,
            AnswerRequest(
                question=payload.question,
                session_id=payload.session_id,
                allow_cloud=payload.allow_cloud,
                allow_web=payload.allow_web,
                deep_research=payload.deep_research,
            ),
        )
        request.state.metric_route = result.decision.route.value
        return answer_response(result, trace_id=_context_from_request(request).trace_id)

    @app.delete(
        "/v1/knowledge-bases/{knowledge_base_id}/sessions/{session_id}",
        response_model=DeleteResponse,
        responses=_error_responses(401, 403, 404, 422, 429, 500, 503),
        tags=["sessions"],
        summary="Delete one conversation session",
    )
    def delete_session(
        request: Request,
        principal: Annotated[Principal, Depends(writer)],
        knowledge_base_id: Annotated[str, Path(pattern=_RESOURCE_PATTERN, max_length=128)],
        session_id: Annotated[str, Path(pattern=_SESSION_PATTERN, max_length=128)],
    ) -> DeleteResponse:
        request.state.operation = "answer"
        consume(request, principal)
        deleted = platform.clear_session(principal, knowledge_base_id, session_id)
        return DeleteResponse(deleted=bool(deleted))

    @app.get(
        "/metrics",
        response_class=Response,
        responses={
            200: {
                "description": "Prometheus text exposition format.",
                "content": {"text/plain": {"schema": {"type": "string"}}},
            },
            401: {"model": ErrorEnvelope},
            403: {"model": ErrorEnvelope},
            429: {"model": ErrorEnvelope},
            500: {"model": ErrorEnvelope},
            503: {"model": ErrorEnvelope},
        },
        tags=["operations"],
        summary="Export Prometheus metrics",
    )
    def metrics(
        request: Request,
        principal: Annotated[Principal, Depends(operator)],
    ) -> Response:
        request.state.operation = "health"
        consume(request, principal)
        payload = platform.metrics.registry.render_prometheus()
        return Response(payload, headers={"Content-Type": _PROMETHEUS_CONTENT_TYPE})

    return app


def _encode_knowledge_base_cursor(record: KnowledgeBaseResponse) -> str:
    payload = json.dumps(
        [record.updated_at, record.id],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_knowledge_base_cursor(cursor: str) -> tuple[float, str]:
    try:
        padding = "=" * (-len(cursor) % 4)
        decoded = base64.b64decode(cursor + padding, altchars=b"-_", validate=True)
        value = json.loads(decoded.decode("ascii"))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise ApiBoundaryError(422, "invalid_request", "Knowledge base cursor is invalid.") from None
    if (
        not isinstance(value, list)
        or len(value) != 2
        or isinstance(value[0], bool)
        or not isinstance(value[0], (int, float))
        or not math.isfinite(value[0])
        or value[0] < 0
        or not isinstance(value[1], str)
        or re.fullmatch(_RESOURCE_PATTERN, value[1]) is None
    ):
        raise ApiBoundaryError(422, "invalid_request", "Knowledge base cursor is invalid.")
    return float(value[0]), value[1]


__all__ = [
    "AnswerPayload",
    "AnswerResponse",
    "DeleteResponse",
    "DocumentResponse",
    "ErrorDetail",
    "ErrorEnvelope",
    "HealthResponse",
    "JobResponse",
    "KnowledgeBaseListResponse",
    "KnowledgeBaseResponse",
    "KnowledgeBaseSubmissionResponse",
    "create_app",
]
