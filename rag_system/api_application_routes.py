"""Published-application HTTP route registration."""

import json

from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, Query, Request
from pydantic import Field

from rag_system.api_contract import (
    ApplicationAnswerPayload,
    ApplicationAnswerResponse,
    ApplicationEvaluationCreatePayload,
    ApplicationEvaluationListResponse,
    ApplicationEvaluationResponse,
    ApplicationCreatePayload,
    ApplicationListResponse,
    ApplicationResponse,
    AuditEventListResponse,
    DeploymentCreatePayload,
    DeploymentListResponse,
    DeploymentResponse,
    DraftResponse,
    DraftRevisionCreatePayload,
    DraftUpdatePayload,
    ProjectCreatePayload,
    ProjectListResponse,
    ProjectResponse,
    RevisionCreatePayload,
    RevisionListResponse,
    RevisionResponse,
    ResourceBindingListResponse,
    audit_event_response,
    answer_response,
    application_response,
    application_evaluation_response,
    binding_response,
    deployment_response,
    draft_response,
    project_response,
    revision_response,
)
from rag_system.api_responses import context_from_request
from rag_system.api_security import ApiSecurityDependencies
from rag_system.application_runtime import ApplicationRuntime
from rag_system.application_contracts import (
    AnswerPolicy,
    KnowledgeChatConfiguration,
    RetrievalProfile,
    SessionPolicy,
)
from rag_system.application_service import ApplicationService
from rag_system.application_service import ApplicationServiceValidationError
from rag_system.application_evaluation import ApplicationEvaluationError, ApplicationEvaluationReport
from rag_system.config import Settings
from rag_system.tenancy import Principal


class ErrorResponses(Protocol):
    def __call__(self, *status_codes: int) -> dict[int, dict[str, object]]: ...


def register_application_routes(
    app: FastAPI,
    *,
    runtime: ApplicationRuntime,
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

    @app.get(
        "/v1/applications/{application_id}/draft",
        response_model=DraftResponse,
        tags=["applications"],
    )
    def get_draft(
        request: Request,
        principal: Annotated[Principal, Depends(security.writer)],
        application_id: str,
    ) -> DraftResponse:
        request.state.operation = "application_read"
        security.consume(request, principal)
        return draft_response(service.get_draft(principal, application_id))

    @app.put(
        "/v1/applications/{application_id}/draft",
        response_model=DraftResponse,
        tags=["applications"],
    )
    def update_draft(
        request: Request,
        principal: Annotated[Principal, Depends(security.writer)],
        application_id: str,
        payload: DraftUpdatePayload,
    ) -> DraftResponse:
        request.state.operation = "application_manage"
        security.consume(request, principal)
        configuration = KnowledgeChatConfiguration(
            knowledge_base_ids=tuple(payload.knowledge_base_ids),
            model_profile_id=payload.model_profile_id,
            retrieval_profile=RetrievalProfile(payload.retrieval_profile),
            answer_policy=AnswerPolicy(**payload.answer_policy.model_dump()),
            session_policy=SessionPolicy(**payload.session_policy.model_dump()),
        )
        return draft_response(
            service.update_knowledge_draft(
                principal,
                application_id,
                configuration,
                expected_version=payload.expected_version,
                change_summary=payload.change_summary,
            )
        )

    @app.post(
        "/v1/applications/{application_id}/draft/revisions",
        response_model=RevisionResponse,
        tags=["applications"],
    )
    def create_revision_from_draft(
        request: Request,
        principal: Annotated[Principal, Depends(security.writer)],
        application_id: str,
        payload: DraftRevisionCreatePayload,
    ) -> RevisionResponse:
        request.state.operation = "application_manage"
        security.consume(request, principal)
        return revision_response(
            service.create_revision_from_draft(
                principal, application_id, expected_version=payload.expected_version
            )
        )

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
            model_profile_id=payload.model_profile_id,
            retrieval_profile=RetrievalProfile(payload.retrieval_profile),
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

    @app.get(
        "/v1/applications/{application_id}/revisions/{revision_id}/bindings",
        response_model=ResourceBindingListResponse,
        tags=["applications"],
    )
    def list_bindings(
        request: Request,
        principal: Annotated[Principal, Depends(security.reader)],
        application_id: str,
        revision_id: str,
    ) -> ResourceBindingListResponse:
        request.state.operation = "application_read"
        security.consume(request, principal)
        items = tuple(
            binding_response(item)
            for item in service.list_bindings(principal, application_id, revision_id)
        )
        return ResourceBindingListResponse(items=items, count=len(items))

    @app.get(
        "/v1/applications/{application_id}/revisions/{revision_id}/evaluations",
        response_model=ApplicationEvaluationListResponse,
        tags=["applications"],
    )
    def list_evaluations(
        request: Request,
        principal: Annotated[Principal, Depends(security.reader)],
        application_id: str,
        revision_id: str,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> ApplicationEvaluationListResponse:
        request.state.operation = "application_read"
        security.consume(request, principal)
        items = tuple(
            application_evaluation_response(item)
            for item in service.list_evaluations(
                principal, application_id, revision_id, limit=limit
            )
        )
        return ApplicationEvaluationListResponse(items=items, count=len(items))

    @app.post(
        "/v1/applications/{application_id}/revisions/{revision_id}/evaluations",
        response_model=ApplicationEvaluationResponse,
        tags=["applications"],
    )
    def record_evaluation(
        request: Request,
        principal: Annotated[Principal, Depends(security.writer)],
        application_id: str,
        revision_id: str,
        payload: ApplicationEvaluationCreatePayload,
    ) -> ApplicationEvaluationResponse:
        request.state.operation = "application_manage"
        security.consume(request, principal)
        try:
            report = ApplicationEvaluationReport.from_json(
                json.dumps(payload.report, ensure_ascii=False, separators=(",", ":"))
            )
        except ApplicationEvaluationError as error:
            raise ApplicationServiceValidationError("Evaluation report is invalid.") from error
        if report.application_id != application_id or report.revision_id != revision_id:
            raise ApplicationServiceValidationError(
                "Evaluation report does not match the requested application revision."
            )
        return application_evaluation_response(service.record_evaluation(principal, report))

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
            service.publish(
                principal,
                application_id,
                payload.revision_id,
                expected_active_revision_id=payload.expected_active_revision_id,
            ).deployment
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

    @app.get(
        "/v1/applications/{application_id}/audit-events",
        response_model=AuditEventListResponse,
        tags=["applications"],
    )
    def list_audit_events(
        request: Request,
        principal: Annotated[Principal, Depends(security.reader)],
        application_id: str,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> AuditEventListResponse:
        request.state.operation = "application_read"
        security.consume(request, principal)
        items = tuple(
            audit_event_response(item)
            for item in service.list_audit_events(
                principal, application_id=application_id, limit=limit
            )
        )
        return AuditEventListResponse(items=items, count=len(items))

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
