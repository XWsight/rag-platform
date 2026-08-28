"""Published-application HTTP route registration."""

from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, Query, Request
from pydantic import Field

from rag_system.api_contract import (
    ApplicationAnswerPayload,
    ApplicationAnswerResponse,
    ApplicationCreatePayload,
    ApplicationListResponse,
    ApplicationResponse,
    DeploymentCreatePayload,
    DeploymentListResponse,
    DeploymentResponse,
    ProjectCreatePayload,
    ProjectListResponse,
    ProjectResponse,
    RevisionCreatePayload,
    RevisionListResponse,
    RevisionResponse,
    answer_response,
    application_response,
    deployment_response,
    project_response,
    revision_response,
)
from rag_system.api_responses import context_from_request
from rag_system.api_security import ApiSecurityDependencies
from rag_system.application_runtime import KnowledgeApplicationRuntime
from rag_system.application_contracts import AnswerPolicy, KnowledgeChatConfiguration, SessionPolicy
from rag_system.application_service import ApplicationService
from rag_system.config import Settings
from rag_system.tenancy import Principal


class ErrorResponses(Protocol):
    def __call__(self, *status_codes: int) -> dict[int, dict[str, object]]: ...


def register_application_routes(
    app: FastAPI,
    *,
    runtime: KnowledgeApplicationRuntime,
    service: ApplicationService,
    settings: Settings,
    security: ApiSecurityDependencies,
    error_responses: ErrorResponses,
) -> None:
    """Expose answers from the active immutable revision of an application."""

    class ConfiguredApplicationAnswerPayload(ApplicationAnswerPayload):
        question: str = Field(min_length=1, max_length=settings.max_question_characters)

    @app.post("/v1/projects", response_model=ProjectResponse, tags=["applications"])
    def create_project(
        request: Request,
        principal: Annotated[Principal, Depends(security.writer)],
        payload: ProjectCreatePayload,
    ) -> ProjectResponse:
        request.state.operation = "application_manage"
        security.consume(request, principal)
        return project_response(
            service.create_project(principal, payload.display_name, payload.description)
        )

    @app.get("/v1/projects", response_model=ProjectListResponse, tags=["applications"])
    def list_projects(
        request: Request,
        principal: Annotated[Principal, Depends(security.reader)],
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> ProjectListResponse:
        request.state.operation = "application_read"
        security.consume(request, principal)
        items = tuple(
            project_response(item) for item in service.list_projects(principal, limit=limit)
        )
        return ProjectListResponse(items=items, count=len(items))

    @app.get("/v1/projects/{project_id}", response_model=ProjectResponse, tags=["applications"])
    def get_project(
        request: Request, principal: Annotated[Principal, Depends(security.reader)], project_id: str
    ) -> ProjectResponse:
        request.state.operation = "application_read"
        security.consume(request, principal)
        return project_response(service.get_project(principal, project_id))

    @app.post("/v1/applications", response_model=ApplicationResponse, tags=["applications"])
    def create_application(
        request: Request,
        principal: Annotated[Principal, Depends(security.writer)],
        payload: ApplicationCreatePayload,
    ) -> ApplicationResponse:
        request.state.operation = "application_manage"
        security.consume(request, principal)
        return application_response(
            service.create_knowledge_application(
                principal, payload.project_id, payload.display_name
            )
        )

    @app.get("/v1/applications", response_model=ApplicationListResponse, tags=["applications"])
    def list_applications(
        request: Request,
        principal: Annotated[Principal, Depends(security.reader)],
        project_id: str,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> ApplicationListResponse:
        request.state.operation = "application_read"
        security.consume(request, principal)
        items = tuple(
            application_response(item)
            for item in service.list_applications(principal, project_id, limit=limit)
        )
        return ApplicationListResponse(items=items, count=len(items))

    @app.get(
        "/v1/applications/{application_id}",
        response_model=ApplicationResponse,
        tags=["applications"],
    )
    def get_application(
        request: Request,
        principal: Annotated[Principal, Depends(security.reader)],
        application_id: str,
    ) -> ApplicationResponse:
        request.state.operation = "application_read"
        security.consume(request, principal)
        return application_response(service.get_application(principal, application_id))

    @app.delete(
        "/v1/applications/{application_id}",
        response_model=ApplicationResponse,
        tags=["applications"],
        summary="Archive an application while retaining its immutable history",
    )
    def archive_application(
        request: Request,
        principal: Annotated[Principal, Depends(security.operator)],
        application_id: str,
    ) -> ApplicationResponse:
        request.state.operation = "application_manage"
        security.consume(request, principal)
        return application_response(service.archive_application(principal, application_id))

    @app.post(
        "/v1/applications/{application_id}/revisions",
        response_model=RevisionResponse,
        tags=["applications"],
    )
    def create_revision(
        request: Request,
        principal: Annotated[Principal, Depends(security.writer)],
        application_id: str,
        payload: RevisionCreatePayload,
    ) -> RevisionResponse:
        request.state.operation = "application_manage"
        security.consume(request, principal)
        configuration = KnowledgeChatConfiguration(
            knowledge_base_ids=tuple(payload.knowledge_base_ids),
            retrieval_profile=payload.retrieval_profile,
            answer_policy=AnswerPolicy(**payload.answer_policy.model_dump()),
            session_policy=SessionPolicy(**payload.session_policy.model_dump()),
        )
        return revision_response(
            service.create_knowledge_revision(
                principal, application_id, configuration, change_summary=payload.change_summary
            )
        )

    @app.get(
        "/v1/applications/{application_id}/revisions",
        response_model=RevisionListResponse,
        tags=["applications"],
    )
    def list_revisions(
        request: Request,
        principal: Annotated[Principal, Depends(security.reader)],
        application_id: str,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> RevisionListResponse:
        request.state.operation = "application_read"
        security.consume(request, principal)
        items = tuple(
            revision_response(item)
            for item in service.list_revisions(principal, application_id, limit=limit)
        )
        return RevisionListResponse(items=items, count=len(items))

    @app.post(
        "/v1/applications/{application_id}/deployments",
        response_model=DeploymentResponse,
        tags=["applications"],
    )
    def create_deployment(
        request: Request,
        principal: Annotated[Principal, Depends(security.operator)],
        application_id: str,
        payload: DeploymentCreatePayload,
    ) -> DeploymentResponse:
        request.state.operation = "application_publish"
        security.consume(request, principal)
        return deployment_response(
            service.publish(principal, application_id, payload.revision_id).deployment
        )

    @app.get(
        "/v1/applications/{application_id}/deployments",
        response_model=DeploymentListResponse,
        tags=["applications"],
    )
    def list_deployments(
        request: Request,
        principal: Annotated[Principal, Depends(security.reader)],
        application_id: str,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> DeploymentListResponse:
        request.state.operation = "application_read"
        security.consume(request, principal)
        items = tuple(
            deployment_response(item)
            for item in service.list_deployments(principal, application_id, limit=limit)
        )
        return DeploymentListResponse(items=items, count=len(items))

    @app.post(
        "/v1/apps/{application_id}/answer",
        response_model=ApplicationAnswerResponse,
        responses=error_responses(401, 403, 404, 409, 422, 429, 500, 503),
        tags=["applications"],
        summary="Answer through a published application revision",
    )
    def answer_application(
        request: Request,
        principal: Annotated[Principal, Depends(security.reader)],
        application_id: str,
        payload: ConfiguredApplicationAnswerPayload,
    ) -> ApplicationAnswerResponse:
        request.state.operation = "application_answer"
        security.consume(request, principal)
        answer = runtime.answer(
            principal, application_id, question=payload.question, session_id=payload.session_id
        )
        request.state.metric_route = answer.result.decision.route.value
        result = answer_response(answer.result, trace_id=context_from_request(request).trace_id)
        return ApplicationAnswerResponse(
            **result.model_dump(),
            application_id=answer.application_id,
            revision_id=answer.revision_id,
        )


__all__ = ["register_application_routes"]
