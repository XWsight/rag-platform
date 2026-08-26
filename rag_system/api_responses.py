"""Trace context and stable safe error envelopes for the HTTP boundary."""

import re

from fastapi import Request
from fastapi.responses import JSONResponse

from rag_system.api_contract import ErrorDetail, ErrorEnvelope
from rag_system.observability import TraceContext


_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._:-]{0,127})?")


def new_request_context(request: Request) -> TraceContext:
    """Create a context using only bounded, safe incoming correlation IDs."""

    return TraceContext.new(
        trace_id=_safe_incoming_identifier(request.headers.get("X-Trace-ID")),
        request_id=_safe_incoming_identifier(request.headers.get("X-Request-ID")),
    )


def context_from_request(request: Request) -> TraceContext:
    context = getattr(request.state, "trace_context", None)
    return context if isinstance(context, TraceContext) else TraceContext.new()


def error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Render the sole public error schema without reflecting unsafe details."""

    context = context_from_request(request)
    envelope = ErrorEnvelope(
        error=ErrorDetail(
            code=code,
            message=message,
            trace_id=context.trace_id,
            request_id=context.request_id,
        )
    )
    safe_headers = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "X-Trace-ID": context.trace_id,
        "X-Request-ID": context.request_id,
        **dict(headers or {}),
    }
    if status_code == 401:
        safe_headers.setdefault("WWW-Authenticate", "Bearer")
    return JSONResponse(status_code=status_code, content=envelope.model_dump(mode="json"), headers=safe_headers)


def _safe_incoming_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized if _IDENTIFIER_PATTERN.fullmatch(normalized) is not None else None


__all__ = ["context_from_request", "error_response", "new_request_context"]
