"""Answer route registration with a configuration-bound request schema."""

from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, Request
from pydantic import Field

from rag_system.api_contract import AnswerPayload, AnswerResponse, answer_response
from rag_system.api_responses import context_from_request
from rag_system.api_security import ApiSecurityDependencies
from rag_system.application import RagApplication
from rag_system.config import Settings
from rag_system.domain import AnswerRequest
from rag_system.tenancy import Principal


class ErrorResponses(Protocol):
    """Build the shared OpenAPI declaration for public error envelopes."""

    def __call__(self, *status_codes: int) -> dict[int, dict[str, object]]: ...


def register_answer_routes(
    app: FastAPI,
    *,
    platform: RagApplication,
    settings: Settings,
    security: ApiSecurityDependencies,
    error_responses: ErrorResponses,
) -> None:
    """Register the answer endpoint using the runtime question length bound."""

    class ConfiguredAnswerPayload(AnswerPayload):
        question: str = Field(min_length=1, max_length=settings.max_question_characters)

    @app.post(
        "/v1/answers",
        response_model=AnswerResponse,
        responses=error_responses(401, 403, 404, 409, 422, 429, 500, 503),
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


__all__ = ["register_answer_routes"]
