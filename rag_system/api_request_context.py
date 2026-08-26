"""Request lifecycle middleware for tracing, upload bounds, and observability."""

import time

from fastapi import FastAPI, Request
from fastapi.responses import Response
from starlette.middleware.base import RequestResponseEndpoint
from starlette.types import Message

from rag_system.api_error_handlers import safe_emit
from rag_system.api_errors import ApiBoundaryError
from rag_system.api_responses import error_response, new_request_context
from rag_system.api_security import ApiSecurityDependencies
from rag_system.application import RagApplication
from rag_system.config import Settings
from rag_system.observability import JsonEventLogger
from rag_system.rate_limit import RateLimitDecision
from rag_system.tenancy import AuthenticationError


def install_request_context_middleware(
    app: FastAPI,
    *,
    platform: RagApplication,
    settings: Settings,
    security: ApiSecurityDependencies,
    logger: JsonEventLogger,
) -> None:
    """Install one middleware that owns all request-scoped boundary state."""

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        started = time.perf_counter()
        context = new_request_context(request)
        request.state.trace_context = context
        request.state.operation = operation_for(request.method, request.url.path)
        request.state.metric_route = "retrieval_only"
        response: Response | None = None
        try:
            if request.method == "POST" and request.url.path == "/v1/knowledge-bases":
                try:
                    principal = security.authenticate_request(request)
                    if not principal.has_role("writer"):
                        raise ApiBoundaryError(
                            403,
                            "forbidden",
                            "The operation is not permitted.",
                        )
                    security.consume(request, principal, tokens=2)
                    request.state.upload_rate_preconsumed = True
                except AuthenticationError:
                    response = error_response(
                        request,
                        status_code=401,
                        code="authentication_failed",
                        message="Authentication failed.",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                    return response
                except ApiBoundaryError as error:
                    response = error_response(
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
                    response = error_response(
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
            outcome = outcome_for(status_code)
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
            safe_emit(logger, "http_request", context=context, fields=fields)
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


def outcome_for(status_code: int) -> str:
    """Map only HTTP status classes to low-cardinality metrics outcomes."""

    if status_code < 400:
        return "success"
    if status_code == 429:
        return "rate_limited"
    if status_code in {401, 403, 404}:
        return "refused"
    if status_code == 503:
        return "unavailable"
    return "error"


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


def _format_rate_value(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.3f}".rstrip("0").rstrip(".")


def operation_for(method: str, path: str) -> str:
    if path.startswith("/health") or path == "/metrics":
        return "health"
    if path == "/v1/answers":
        return "answer"
    if method == "POST" and path == "/v1/knowledge-bases":
        return "ingest"
    return "index"


__all__ = ["install_request_context_middleware", "outcome_for"]
