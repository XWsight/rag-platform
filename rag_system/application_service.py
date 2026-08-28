"""Application use cases over storage-neutral platform ports."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rag_system.application_contracts import (
    APPLICATION_CONFIGURATION_SCHEMA_VERSION,
    Application,
    ApplicationAuditEventType,
    ApplicationKind,
    ApplicationRevision,
    ApplicationStatus,
    AuditEvent,
    Deployment,
    DeploymentEnvironment,
    KnowledgeChatConfiguration,
    Project,
    ResourceAccessMode,
    ResourceBinding,
    ResourceKind,
    is_valid_timestamp,
)
from rag_system.application_ports import ApplicationRepository, KnowledgeBaseRepository
from rag_system.knowledge_base_contracts import KnowledgeBaseStatus
from rag_system.tenancy import Principal


class ApplicationServiceError(Exception):
    """Base error for application use-case failures."""


class ApplicationAuthorizationError(ApplicationServiceError):
    def __init__(self) -> None:
        super().__init__("The operation is not permitted.")


class ApplicationResourceUnavailableError(ApplicationServiceError):
    def __init__(self) -> None:
        super().__init__("A required application resource is unavailable.")


class ApplicationServiceValidationError(ApplicationServiceError, ValueError):
    """An application command does not satisfy the platform contract."""


@dataclass(frozen=True, slots=True)
class PublishedApplication:
    application: Application
    deployment: Deployment


class ApplicationService:
    """Creates immutable revisions and atomically publishes a selected revision."""

    def __init__(
        self,
        repository: ApplicationRepository,
        knowledge_bases: KnowledgeBaseRepository,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._repository = repository
        self._knowledge_bases = knowledge_bases
        self._clock = clock

    def create_project(
        self, principal: Principal, display_name: str, description: str = ""
    ) -> Project:
        _require_writer(principal)
        now = self._now()
        project = Project(
            project_id=_new_id("prj"),
            tenant_id=principal.tenant_id,
            display_name=display_name,
            description=description,
            created_at=now,
            updated_at=now,
        )
        created = self._repository.create_project(principal, project)
        self._repository.record_audit_event(
            principal,
            AuditEvent(
                audit_event_id=_new_id("audit"),
                tenant_id=principal.tenant_id,
                event_type=ApplicationAuditEventType.PROJECT_CREATED,
                occurred_at=now,
                actor=principal.subject,
                summary="Created a project.",
                project_id=created.project_id,
            ),
        )
        return created

    def get_project(self, principal: Principal, project_id: str) -> Project:
        _require_reader(principal)
        return self._repository.get_project(principal, project_id)

    def list_projects(self, principal: Principal, *, limit: int = 50) -> tuple[Project, ...]:
        _require_reader(principal)
        return self._repository.list_projects(principal, limit=limit)

    def create_knowledge_application(
        self, principal: Principal, project_id: str, display_name: str
    ) -> Application:
        _require_writer(principal)
        now = self._now()
        application = Application(
            application_id=_new_id("app"),
            tenant_id=principal.tenant_id,
            project_id=project_id,
            display_name=display_name,
            application_kind=ApplicationKind.KNOWLEDGE_CHAT,
            active_revision_id=None,
            status=ApplicationStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
        created = self._repository.create_application(principal, application)
        self._repository.record_audit_event(
            principal,
            AuditEvent(
                audit_event_id=_new_id("audit"),
                tenant_id=principal.tenant_id,
                event_type=ApplicationAuditEventType.APPLICATION_CREATED,
                occurred_at=now,
                actor=principal.subject,
                summary="Created a knowledge-chat application.",
                project_id=created.project_id,
                application_id=created.application_id,
            ),
        )
        return created

    def get_application(self, principal: Principal, application_id: str) -> Application:
        _require_reader(principal)
        return self._repository.get_application(principal, application_id)

    def list_applications(
        self, principal: Principal, project_id: str, *, limit: int = 50
    ) -> tuple[Application, ...]:
        _require_reader(principal)
        return self._repository.list_applications(principal, project_id, limit=limit)

    def archive_application(self, principal: Principal, application_id: str) -> Application:
        _require_operator(principal)
        application = self._repository.get_application(principal, application_id)
        if application.status is ApplicationStatus.ARCHIVED:
            raise ApplicationServiceValidationError("Application is already archived.")
        now = self._now()
        event = AuditEvent(
            audit_event_id=_new_id("audit"), tenant_id=principal.tenant_id,
            event_type=ApplicationAuditEventType.APPLICATION_ARCHIVED, occurred_at=now,
            actor=principal.subject, summary="Archived an application.",
            project_id=application.project_id, application_id=application.application_id,
        )
        return self._repository.archive_application(
            principal, application.application_id, event, updated_at=now
        )

    def create_knowledge_revision(
        self,
        principal: Principal,
        application_id: str,
        configuration: KnowledgeChatConfiguration,
        *,
        change_summary: str,
    ) -> ApplicationRevision:
        _require_writer(principal)
        application = self._repository.get_application(principal, application_id)
        if application.status is ApplicationStatus.ARCHIVED:
            raise ApplicationServiceValidationError(
                "Archived applications cannot accept new revisions."
            )
        if application.application_kind is not ApplicationKind.KNOWLEDGE_CHAT:
            raise ApplicationServiceValidationError("Unsupported application kind.")
        self._verify_ready_knowledge_bases(principal, configuration.knowledge_base_ids)
        now = self._now()
        revisions = self._repository.list_revisions(
            principal, application.application_id, limit=100
        )
        revision = ApplicationRevision(
            revision_id=_new_id("rev"),
            application_id=application.application_id,
            revision_number=(max((item.revision_number for item in revisions), default=0) + 1),
            configuration_schema_version=APPLICATION_CONFIGURATION_SCHEMA_VERSION,
            configuration=configuration,
            created_at=now,
            created_by=principal.subject,
            change_summary=change_summary,
        )
        bindings = tuple(
            ResourceBinding(
                binding_id=_new_id("bind"),
                application_id=application.application_id,
                revision_id=revision.revision_id,
                resource_kind=ResourceKind.KNOWLEDGE_BASE,
                resource_id=resource_id,
                access_mode=ResourceAccessMode.READ,
                created_at=now,
            )
            for resource_id in configuration.knowledge_base_ids
        )
        created = self._repository.create_revision(principal, revision, bindings)
        self._repository.record_audit_event(
            principal,
            AuditEvent(
                audit_event_id=_new_id("audit"),
                tenant_id=principal.tenant_id,
                event_type=ApplicationAuditEventType.REVISION_CREATED,
                occurred_at=now,
                actor=principal.subject,
                summary="Created an immutable application revision.",
                project_id=application.project_id,
                application_id=application.application_id,
                revision_id=created.revision_id,
            ),
        )
        return created

    def list_revisions(
        self, principal: Principal, application_id: str, *, limit: int = 50
    ) -> tuple[ApplicationRevision, ...]:
        _require_reader(principal)
        return self._repository.list_revisions(principal, application_id, limit=limit)

    def list_deployments(
        self, principal: Principal, application_id: str, *, limit: int = 50
    ) -> tuple[Deployment, ...]:
        _require_reader(principal)
        return self._repository.list_deployments(principal, application_id, limit=limit)

    def publish(
        self,
        principal: Principal,
        application_id: str,
        revision_id: str,
        *,
        environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION,
    ) -> PublishedApplication:
        _require_operator(principal)
        if environment is not DeploymentEnvironment.PRODUCTION:
            raise ApplicationServiceValidationError("Only production deployments are supported.")
        application = self._repository.get_application(principal, application_id)
        if application.status is ApplicationStatus.ARCHIVED:
            raise ApplicationServiceValidationError(
                "Archived applications cannot be published."
            )
        revision = self._repository.get_revision(principal, application.application_id, revision_id)
        self._verify_ready_knowledge_bases(principal, revision.configuration.knowledge_base_ids)
        now = self._now()
        deployment = Deployment(
            deployment_id=_new_id("dep"),
            application_id=application.application_id,
            revision_id=revision.revision_id,
            environment=environment,
            deployed_at=now,
            deployed_by=principal.subject,
        )
        event = AuditEvent(
            audit_event_id=_new_id("audit"),
            tenant_id=principal.tenant_id,
            event_type=ApplicationAuditEventType.DEPLOYMENT_CREATED,
            occurred_at=now,
            actor=principal.subject,
            summary="Published an application revision.",
            project_id=application.project_id,
            application_id=application.application_id,
            revision_id=revision.revision_id,
        )
        active = self._repository.publish(principal, deployment, event, updated_at=now)
        return PublishedApplication(application=active, deployment=deployment)

    def rollback(
        self, principal: Principal, application_id: str, revision_id: str
    ) -> PublishedApplication:
        return self.publish(principal, application_id, revision_id)

    def _verify_ready_knowledge_bases(
        self, principal: Principal, resource_ids: Sequence[str]
    ) -> None:
        for resource_id in resource_ids:
            try:
                record = self._knowledge_bases.get(principal, resource_id)
            except Exception as error:
                raise ApplicationResourceUnavailableError() from error
            if record.status is not KnowledgeBaseStatus.READY:
                raise ApplicationResourceUnavailableError()

    def _now(self) -> float:
        value = float(self._clock())
        if not is_valid_timestamp(value):
            raise ApplicationServiceValidationError("clock returned an invalid timestamp.")
        return value


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(16)}"


def _require_writer(principal: Principal) -> None:
    if not isinstance(principal, Principal) or not principal.has_role("writer"):
        raise ApplicationAuthorizationError()


def _require_reader(principal: Principal) -> None:
    if not isinstance(principal, Principal) or not principal.has_role("reader"):
        raise ApplicationAuthorizationError()


def _require_operator(principal: Principal) -> None:
    if not isinstance(principal, Principal) or not principal.has_role("operator"):
        raise ApplicationAuthorizationError()


__all__ = [
    "ApplicationAuthorizationError",
    "ApplicationResourceUnavailableError",
    "ApplicationService",
    "ApplicationServiceError",
    "ApplicationServiceValidationError",
    "PublishedApplication",
]
