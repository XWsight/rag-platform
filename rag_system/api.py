"""Authenticated, tenant-scoped HTTP boundary for the production service."""

import base64
import hashlib
import json
import logging
import math
import re
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, Header, Path, Query, Request, Security, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import RequestResponseEndpoint
from starlette.types import Message

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
from rag_system.api_errors import (
    APPLICATION_ERROR_TYPES,
    ApiBoundaryError,
    classify_application_error,
    classify_http_error,
)
from rag_system.application import RagApplication
from rag_system.api_uploads import read_uploads as _read_uploads
from rag_system.domain import AnswerRequest
from rag_system.api_responses import (
    context_from_request as _context_from_request,
    error_response as _error_response,
    new_request_context as _request_context,
)
from rag_system.observability import JsonEventLogger, TraceContext
from rag_system.rate_limit import RateLimitDecision, TokenBucketRateLimiter
from rag_system.tenancy import (
    ApiKeyAuthenticator,
    AuthenticationError,
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

    api_key_scheme = APIKeyHeader(
        name="X-API-Key",
        scheme_name="ApiKeyAuth",
        description="A tenant-scoped service API key.",
        auto_error=False,
    )
    bearer_scheme = HTTPBearer(
        scheme_name="BearerAuth",
        bearerFormat="API key",
        description="The same tenant API key carried as a Bearer credential.",
        auto_error=False,
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        started = time.perf_counter()
        context = _request_context(request)
        request.state.trace_context = context
        request.state.operation = _operation_for(request.method, request.url.path)
        request.state.metric_route = "retrieval_only"
        response: Response | None = None
        try:
            if request.method == "POST" and request.url.path == "/v1/knowledge-bases":
                try:
                    principal = authenticator.authenticate_headers(_raw_headers(request))
                    if not principal.has_role("writer"):
                        raise ApiBoundaryError(
                            403,
                            "forbidden",
                            "The operation is not permitted.",
                        )
                    request.state.principal = principal
                    request.state.tenant_hash = hashlib.sha256(
                        principal.tenant_id.value.encode("utf-8")
                    ).hexdigest()[:16]
                    consume(request, principal, tokens=2)
                    request.state.upload_rate_preconsumed = True
                except AuthenticationError:
                    response = _error_response(
                        request,
                        status_code=401,
                        code="authentication_failed",
                        message="Authentication failed.",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                    return response
                except ApiBoundaryError as error:
                    response = _error_response(
                        request,
                        status_code=error.status_code,
                        code=error.code,
                        message=error.safe_message,
                        headers=error.headers,
                    )
                    return response
                body_limit = _multipart_body_limit(
                    settings.max_total_bytes,
                    settings.max_documents,
                )
                header_error = _content_length_error(request, body_limit)
                if header_error is not None:
                    response = _error_response(
                        request,
                        status_code=header_error.status_code,
                        code=header_error.code,
                        message=header_error.safe_message,
                    )
                    return response
                _install_receive_limit(request, body_limit)
            response = await call_next(request)
            return response
        finally:
            status_code = response.status_code if response is not None else 500
            duration = max(0.0, time.perf_counter() - started)
            outcome = _outcome_for(status_code)
            operation = request.state.operation
            metric_route = request.state.metric_route
            _record_request_metrics(platform, operation, outcome, metric_route, duration)
            fields: dict[str, object] = {
                "operation": operation,
                "outcome": outcome,
                "http_status": status_code,
                "duration_ms": duration * 1_000,
            }
            tenant_hash = getattr(request.state, "tenant_hash", "")
            if tenant_hash:
                fields["tenant_hash"] = tenant_hash
            _safe_emit(logger, "http_request", context=context, fields=fields)
            if response is not None:
                response.headers["X-Trace-ID"] = context.trace_id
                response.headers["X-Request-ID"] = context.request_id
                response.headers["X-Content-Type-Options"] = "nosniff"
                response.headers["Cache-Control"] = "no-store"
                decision = getattr(request.state, "rate_limit_decision", None)
                if isinstance(decision, RateLimitDecision):
                    response.headers["X-RateLimit-Limit"] = _format_rate_value(decision.capacity)
                    response.headers["X-RateLimit-Remaining"] = _format_rate_value(
                        decision.remaining_tokens
                    )

    @app.exception_handler(ApiBoundaryError)
    async def api_boundary_handler(request: Request, error: ApiBoundaryError) -> JSONResponse:
        return _error_response(
            request,
            status_code=error.status_code,
            code=error.code,
            message=error.safe_message,
            headers=error.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, _: RequestValidationError) -> JSONResponse:
        return _error_response(
            request,
            status_code=422,
            code="invalid_request",
            message="The request could not be validated.",
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, error: StarletteHTTPException) -> JSONResponse:
        code, message = classify_http_error(error.status_code)
        return _error_response(
            request,
            status_code=error.status_code,
            code=code,
            message=message,
        )

    async def application_error_handler(request: Request, error: Exception) -> JSONResponse:
        status_code, code, message = classify_application_error(error)
        level = logging.ERROR if status_code >= 500 else logging.WARNING
        _safe_emit(
            logger,
            "application_error",
            context=_context_from_request(request),
            fields={"operation": request.state.operation, "outcome": _outcome_for(status_code), "error_type": code},
            level=level,
        )
        return _error_response(
            request,
            status_code=status_code,
            code=code,
            message=message,
        )

    for error_type in APPLICATION_ERROR_TYPES:
        app.add_exception_handler(error_type, application_error_handler)
    app.add_exception_handler(Exception, application_error_handler)

    async def authenticate(
        request: Request,
        _api_key: Annotated[str | None, Security(api_key_scheme)],
        _bearer: Annotated[HTTPAuthorizationCredentials | None, Security(bearer_scheme)],
    ) -> Principal:
        del _api_key, _bearer
        cached = getattr(request.state, "principal", None)
        if isinstance(cached, Principal):
            return cached
        principal = authenticator.authenticate_headers(_raw_headers(request))
        request.state.principal = principal
        request.state.tenant_hash = hashlib.sha256(
            principal.tenant_id.value.encode("utf-8")
        ).hexdigest()[:16]
        return principal

    def require_role(role: Literal["reader", "writer", "operator"]) -> Callable[..., object]:
        async def dependency(
            principal: Annotated[Principal, Depends(authenticate)],
        ) -> Principal:
            if not principal.has_role(role):
                raise ApiBoundaryError(403, "forbidden", "The operation is not permitted.")
            return principal

        return dependency

    reader = require_role("reader")
    writer = require_role("writer")
    operator = require_role("operator")

    def consume(request: Request, principal: Principal, *, tokens: float = 1.0) -> None:
        if getattr(request.state, "upload_rate_preconsumed", False):
            request.state.upload_rate_preconsumed = False
            return
        requested = min(float(tokens), rate_limiter.capacity)
        decision = rate_limiter.acquire(principal.tenant_id.value, tokens=requested)
        request.state.rate_limit_decision = decision
        if not decision.allowed:
            retry_after = max(1, math.ceil(decision.retry_after_seconds))
            raise ApiBoundaryError(
                429,
                "rate_limit_exceeded",
                "The request rate limit was exceeded.",
                headers={"Retry-After": str(retry_after)},
            )

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


def _multipart_body_limit(max_total_bytes: int, max_documents: int) -> int:
    # File names, content-disposition fields, and boundaries are bounded
    # separately from the aggregate document bytes.
    return max_total_bytes + max_documents * 64 * 1024 + 256 * 1024


def _content_length_error(request: Request, body_limit: int) -> ApiBoundaryError | None:
    values = [
        value.decode("latin-1").strip()
        for name, value in request.scope.get("headers", ())
        if name.lower() == b"content-length"
    ]
    transfer_encoding = any(
        name.lower() == b"transfer-encoding"
        for name, _ in request.scope.get("headers", ())
    )
    if len(values) > 1 or (values and transfer_encoding):
        return ApiBoundaryError(400, "invalid_request", "The request could not be validated.")
    if not values:
        return None
    try:
        size = int(values[0], 10)
    except ValueError:
        return ApiBoundaryError(400, "invalid_request", "The request could not be validated.")
    if size < 0:
        return ApiBoundaryError(400, "invalid_request", "The request could not be validated.")
    if size > body_limit:
        return ApiBoundaryError(
            413,
            "upload_limit_exceeded",
            "The upload exceeds the configured limits.",
        )
    return None


def _install_receive_limit(request: Request, body_limit: int) -> None:
    original_receive = request._receive
    received = 0

    async def limited_receive() -> Message:
        nonlocal received
        message = await original_receive()
        if message.get("type") == "http.request":
            body = message.get("body", b"")
            received += len(body)
            if received > body_limit:
                raise ApiBoundaryError(
                    413,
                    "upload_limit_exceeded",
                    "The upload exceeds the configured limits.",
                )
        return message

    request._receive = limited_receive


def _raw_headers(request: Request) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for name, value in request.scope.get("headers", ()):
        result.append((name.decode("latin-1"), value.decode("latin-1")))
    return tuple(result)


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


def _operation_for(method: str, path: str) -> str:
    if path.startswith("/health") or path == "/metrics":
        return "health"
    if path == "/v1/answers":
        return "answer"
    if method == "POST" and path == "/v1/knowledge-bases":
        return "ingest"
    return "index"


def _outcome_for(status_code: int) -> str:
    if status_code < 400:
        return "success"
    if status_code == 429:
        return "rate_limited"
    if status_code in {401, 403, 404}:
        return "refused"
    if status_code == 503:
        return "unavailable"
    return "error"


def _record_request_metrics(
    platform: RagApplication,
    operation: str,
    outcome: str,
    route: str,
    duration_seconds: float,
) -> None:
    try:
        platform.metrics.requests_total.increment(
            labels={"operation": operation, "outcome": outcome}
        )
        platform.metrics.request_duration_seconds.observe(
            duration_seconds,
            labels={"operation": operation, "route": route},
        )
    except Exception:
        return


def _safe_emit(
    logger: JsonEventLogger,
    event: str,
    *,
    context: TraceContext,
    fields: dict[str, object],
    level: int = logging.INFO,
) -> None:
    try:
        logger.emit(event, context=context, fields=fields, level=level)
    except Exception:
        return


def _format_rate_value(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.3f}".rstrip("0").rstrip(".")


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
