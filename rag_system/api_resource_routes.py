"""Knowledge-base, job, session, health, and operations route registration."""

import base64
import json
import math
import re
from collections.abc import Callable
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, File, Form, Header, Path, Query, Request, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from rag_system.api_contract import (
    RESOURCE_PATTERN,
    SESSION_PATTERN,
    DeleteResponse,
    ErrorEnvelope,
    HealthResponse,
    JobResponse,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
    KnowledgeBaseSubmissionResponse,
    job_response,
    knowledge_base_response,
)
from rag_system.api_errors import ApiBoundaryError
from rag_system.api_security import ApiSecurityDependencies
from rag_system.api_uploads import read_uploads
from rag_system.application import RagApplication
from rag_system.config import Settings
from rag_system.tenancy import Principal


_PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
ReadinessCheck = Callable[[], bool]


class ErrorResponses(Protocol):
    """Build the shared OpenAPI declaration for public error envelopes."""

    def __call__(self, *status_codes: int) -> dict[int, dict[str, object]]: ...


def register_resource_routes(
    app: FastAPI,
    *,
    platform: RagApplication,
    settings: Settings,
    security: ApiSecurityDependencies,
    readiness: ReadinessCheck | bool | None,
    error_responses: ErrorResponses,
) -> None:
    """Register non-answer endpoints against explicit application dependencies."""

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
        responses=error_responses(500, 503),
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
        responses=error_responses(401, 403, 409, 413, 422, 429, 500, 503),
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
        uploads = await read_uploads(
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
        responses=error_responses(401, 403, 422, 429, 500, 503),
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
        responses=error_responses(401, 403, 404, 422, 429, 500, 503),
        tags=["knowledge bases"],
        summary="Get a tenant-owned knowledge base",
    )
    def get_knowledge_base(
        request: Request,
        principal: Annotated[Principal, Depends(reader)],
        knowledge_base_id: Annotated[str, Path(pattern=RESOURCE_PATTERN, max_length=128)],
    ) -> KnowledgeBaseResponse:
        request.state.operation = "index"
        consume(request, principal)
        return knowledge_base_response(platform.get_knowledge_base(principal, knowledge_base_id))

    @app.delete(
        "/v1/knowledge-bases/{knowledge_base_id}",
        response_model=DeleteResponse,
        responses=error_responses(401, 403, 404, 409, 422, 429, 500, 503),
        tags=["knowledge bases"],
        summary="Delete a knowledge base and its stored data",
    )
    def delete_knowledge_base(
        request: Request,
        principal: Annotated[Principal, Depends(writer)],
        knowledge_base_id: Annotated[str, Path(pattern=RESOURCE_PATTERN, max_length=128)],
    ) -> DeleteResponse:
        request.state.operation = "index"
        consume(request, principal, tokens=2)
        deleted = platform.delete_knowledge_base(principal, knowledge_base_id)
        return DeleteResponse(deleted=bool(deleted))

    @app.get(
        "/v1/jobs/{job_id}",
        response_model=JobResponse,
        responses=error_responses(401, 403, 404, 422, 429, 500, 503),
        tags=["jobs"],
        summary="Get background job state",
    )
    def get_job(
        request: Request,
        principal: Annotated[Principal, Depends(reader)],
        job_id: Annotated[str, Path(pattern=RESOURCE_PATTERN, max_length=128)],
    ) -> JobResponse:
        request.state.operation = "index"
        consume(request, principal)
        return job_response(platform.get_job(principal, job_id))

    @app.delete(
        "/v1/jobs/{job_id}",
        response_model=JobResponse,
        responses=error_responses(401, 403, 404, 409, 422, 429, 500, 503),
        tags=["jobs"],
        summary="Request cooperative job cancellation",
    )
    def cancel_job(
        request: Request,
        principal: Annotated[Principal, Depends(writer)],
        job_id: Annotated[str, Path(pattern=RESOURCE_PATTERN, max_length=128)],
    ) -> JobResponse:
        request.state.operation = "index"
        consume(request, principal)
        return job_response(platform.cancel_job(principal, job_id))

    @app.delete(
        "/v1/knowledge-bases/{knowledge_base_id}/sessions/{session_id}",
        response_model=DeleteResponse,
        responses=error_responses(401, 403, 404, 422, 429, 500, 503),
        tags=["sessions"],
        summary="Delete one conversation session",
    )
    def delete_session(
        request: Request,
        principal: Annotated[Principal, Depends(writer)],
        knowledge_base_id: Annotated[str, Path(pattern=RESOURCE_PATTERN, max_length=128)],
        session_id: Annotated[str, Path(pattern=SESSION_PATTERN, max_length=128)],
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
        refresh_metrics = getattr(platform, "refresh_operational_metrics", None)
        if callable(refresh_metrics):
            refresh_metrics()
        payload = platform.metrics.registry.render_prometheus()
        return Response(payload, headers={"Content-Type": _PROMETHEUS_CONTENT_TYPE})


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
        or re.fullmatch(RESOURCE_PATTERN, value[1]) is None
    ):
        raise ApiBoundaryError(422, "invalid_request", "Knowledge base cursor is invalid.")
    return float(value[0]), value[1]


__all__ = ["register_resource_routes"]
