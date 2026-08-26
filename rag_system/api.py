"""Authenticated, tenant-scoped HTTP composition root for the production service."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from rag_system.api_contract import (
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
)
from rag_system.api_error_handlers import install_error_handlers
from rag_system.api_openapi import install_multipart_openapi_schema
from rag_system.api_request_context import install_request_context_middleware, outcome_for
from rag_system.api_resource_routes import register_resource_routes
from rag_system.api_responses import context_from_request
from rag_system.api_security import build_api_security_dependencies
from rag_system.application import RagApplication
from rag_system.domain import AnswerRequest
from rag_system.observability import JsonEventLogger
from rag_system.rate_limit import TokenBucketRateLimiter
from rag_system.tenancy import ApiKeyAuthenticator, Principal
from rag_system.web_ui import mount_web_ui


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

    class ConfiguredAnswerPayload(AnswerPayload):
        question: str = Field(min_length=1, max_length=settings.max_question_characters)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if close_on_shutdown:
            await run_in_threadpool(shutdown or platform.close)

    app = FastAPI(
        title=f"{settings.product_name} API",
        summary="Tenant-isolated, grounded knowledge retrieval and answering.",
        version="2.0.0",
        docs_url="/docs" if settings.api_docs_enabled else None,
        redoc_url="/redoc" if settings.api_docs_enabled else None,
        openapi_url="/openapi.json" if settings.api_docs_enabled else None,
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
    install_error_handlers(app, logger=logger, outcome_for=outcome_for)
    register_resource_routes(
        app,
        platform=platform,
        settings=settings,
        security=security,
        readiness=readiness,
        error_responses=_error_responses,
    )

    @app.post(
        "/v1/answers",
        response_model=AnswerResponse,
        responses=_error_responses(401, 403, 404, 409, 422, 429, 500, 503),
        tags=["answers"],
        summary="Answer from a tenant-owned knowledge base",
    )
    def answer(
        request: Request,
        principal: Annotated[Principal, Depends(security.reader)],
        payload: ConfiguredAnswerPayload,
    ) -> AnswerResponse:
        request.state.operation = "research" if payload.deep_research else "answer"
        security.consume(request, principal, tokens=3 if payload.deep_research else 1)
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
        return answer_response(result, trace_id=context_from_request(request).trace_id)

    return app


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
