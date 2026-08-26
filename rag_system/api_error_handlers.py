"""Safe exception-to-HTTP handling for the FastAPI boundary."""

from collections.abc import Callable
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag_system.api_errors import (
    APPLICATION_ERROR_TYPES,
    ApiBoundaryError,
    classify_application_error,
    classify_http_error,
)
from rag_system.api_responses import context_from_request, error_response
from rag_system.observability import JsonEventLogger, TraceContext


def install_error_handlers(
    app: FastAPI,
    *,
    logger: JsonEventLogger,
    outcome_for: Callable[[int], str],
) -> None:
    """Install the sole public error vocabulary for every HTTP failure path."""

    @app.exception_handler(ApiBoundaryError)
    async def api_boundary_handler(request: Request, error: ApiBoundaryError) -> JSONResponse:
        return error_response(
            request,
            status_code=error.status_code,
            code=error.code,
            message=error.safe_message,
            headers=error.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, _: RequestValidationError) -> JSONResponse:
        return error_response(
            request,
            status_code=422,
            code="invalid_request",
            message="The request could not be validated.",
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, error: StarletteHTTPException) -> JSONResponse:
        code, message = classify_http_error(error.status_code)
        return error_response(
            request,
            status_code=error.status_code,
            code=code,
            message=message,
        )

    async def application_error_handler(request: Request, error: Exception) -> JSONResponse:
        status_code, code, message = classify_application_error(error)
        level = logging.ERROR if status_code >= 500 else logging.WARNING
        safe_emit(
            logger,
            "application_error",
            context=context_from_request(request),
            fields={
                "operation": getattr(request.state, "operation", "unknown"),
                "outcome": outcome_for(status_code),
                "error_type": code,
            },
            level=level,
        )
        return error_response(
            request,
            status_code=status_code,
            code=code,
            message=message,
        )

    for error_type in APPLICATION_ERROR_TYPES:
        app.add_exception_handler(error_type, application_error_handler)
    app.add_exception_handler(Exception, application_error_handler)


def safe_emit(
    logger: JsonEventLogger,
    event: str,
    *,
    context: TraceContext,
    fields: dict[str, object],
    level: int = logging.INFO,
) -> None:
    """Keep observability failures from changing the client-visible outcome."""

    try:
        logger.emit(event, context=context, fields=fields, level=level)
    except Exception:
        return


__all__ = ["install_error_handlers", "safe_emit"]
