"""Authorised lifecycle commands for versioned, deployable workflows."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rag_system.application_contracts import is_valid_timestamp
from rag_system.application_ports import ApplicationRepository, KnowledgeBaseRepository
from rag_system.knowledge_base_contracts import KnowledgeBaseStatus
from rag_system.tenancy import Principal
from rag_system.workflow_contracts import WorkflowResourceKind, WorkflowSpec
from rag_system.workflow_models import (
    ApprovalDecision,
    ExecutionBudget,
    Workflow,
    WorkflowDeployment,
    WorkflowDraft,
    WorkflowEvaluation,
    WorkflowRevision,
    WorkflowStatus,
)
from rag_system.workflow_store import WorkflowStore


class WorkflowServiceError(Exception):
    """Base class for public workflow-management failures."""


class WorkflowAuthorizationError(WorkflowServiceError):
    def __init__(self) -> None:
        super().__init__("The operation is not permitted.")


class WorkflowServiceValidationError(WorkflowServiceError, ValueError):
    """A requested workflow change violates its lifecycle contract."""


class WorkflowResourceUnavailableError(WorkflowServiceError):
    def __init__(self) -> None:
        super().__init__("A required workflow resource is unavailable.")


@dataclass(frozen=True, slots=True)
class PublishedWorkflow:
    workflow: Workflow
    deployment: WorkflowDeployment


class WorkflowService:
    """Keep mutable drafts, immutable revisions, checks, and publishing separate."""

    def __init__(
        self,
        store: WorkflowStore,
        projects: ApplicationRepository,
        knowledge_bases: KnowledgeBaseRepository,
        *,
        clock: Callable[[], float] = time.time,
        trusted_model_profile_ids: Sequence[str] = ("default",),
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        profiles = frozenset(trusted_model_profile_ids)
        if not profiles or any(not isinstance(value, str) or not value for value in profiles):
            raise ValueError("trusted_model_profile_ids are invalid")
        self._store = store
        self._projects = projects
        self._knowledge_bases = knowledge_bases
        self._clock = clock
        self._trusted_model_profile_ids = profiles

    def create_workflow(self, principal: Principal, project_id: str, display_name: str) -> Workflow:
        _require_writer(principal)
        # A workflow cannot become an orphaned tenant resource.
        self._projects.get_project(principal, project_id)
        now = self._now()
        return self._store.create_workflow(
            principal,
            Workflow(
                workflow_id=_new_id("wf"), tenant_id=principal.tenant_id, project_id=project_id,
                display_name=display_name, active_revision_id=None, status=WorkflowStatus.ACTIVE,
                created_at=now, updated_at=now,
            ),
        )

    def get_draft(self, principal: Principal, workflow_id: str) -> WorkflowDraft:
        _require_writer(principal)
        return self._store.get_draft(principal, workflow_id)

    def update_draft(
        self, principal: Principal, workflow_id: str, specification: WorkflowSpec, budget: ExecutionBudget,
        *, expected_version: int, change_summary: str,
    ) -> WorkflowDraft:
        _require_writer(principal)
        if isinstance(expected_version, bool) or not isinstance(expected_version, int) or expected_version < 0:
            raise WorkflowServiceValidationError("expected draft version is invalid")
        workflow = self._store.get_workflow(principal, workflow_id)
        if workflow.status is WorkflowStatus.ARCHIVED:
            raise WorkflowServiceValidationError("Archived workflows cannot update drafts.")
        self._verify_resources(principal, specification)
        return self._store.update_draft(
            principal,
            WorkflowDraft(
                workflow_id=workflow.workflow_id, version=expected_version + 1,
                specification=specification, budget=budget, updated_at=self._now(),
                updated_by=principal.subject, change_summary=change_summary,
            ),
            expected_version=expected_version,
        )

    def create_revision_from_draft(
        self, principal: Principal, workflow_id: str, *, expected_version: int
    ) -> WorkflowRevision:
        _require_writer(principal)
        draft = self._store.get_draft(principal, workflow_id)
        if draft.version != expected_version:
            raise WorkflowServiceValidationError("Workflow draft has changed.")
        if draft.specification is None or draft.budget is None:
            raise WorkflowServiceValidationError("Workflow draft has not been configured.")
        return self.create_revision(
            principal, workflow_id, draft.specification, draft.budget,
            change_summary=draft.change_summary,
        )

    def create_revision(
        self, principal: Principal, workflow_id: str, specification: WorkflowSpec,
                        budget: ExecutionBudget, *, change_summary: str) -> WorkflowRevision:
        _require_writer(principal)
        workflow = self._store.get_workflow(principal, workflow_id)
        if workflow.status is WorkflowStatus.ARCHIVED:
            raise WorkflowServiceValidationError("Archived workflows cannot accept revisions.")
        self._verify_resources(principal, specification)
        revisions = self._store.list_revisions(principal, workflow.workflow_id, limit=100)
        return self._store.create_revision(
            principal,
            WorkflowRevision(
                revision_id=_new_id("wfr"), workflow_id=workflow.workflow_id,
                revision_number=max((item.revision_number for item in revisions), default=0) + 1,
                specification=specification, budget=budget, created_at=self._now(),
                created_by=principal.subject, change_summary=change_summary,
            ),
        )

    def record_evaluation(self, principal: Principal, evaluation: WorkflowEvaluation) -> WorkflowEvaluation:
        _require_writer(principal)
        revision = self._store.get_revision(principal, evaluation.workflow_id, evaluation.revision_id)
        if revision.specification_digest != evaluation.specification_digest:
            raise WorkflowServiceValidationError("Evaluation does not match the immutable revision.")
        return self._store.save_evaluation(principal, evaluation)

    def publish(
        self, principal: Principal, workflow_id: str, revision_id: str, *,
        expected_active_revision_id: str | None,
    ) -> PublishedWorkflow:
        _require_operator(principal)
        workflow = self._store.get_workflow(principal, workflow_id)
        if workflow.status is WorkflowStatus.ARCHIVED:
            raise WorkflowServiceValidationError("Archived workflows cannot be published.")
        revision = self._store.get_revision(principal, workflow.workflow_id, revision_id)
        self._verify_resources(principal, revision.specification)
        evaluations = self._store.list_evaluations(principal, workflow.workflow_id, revision.revision_id)
        if not any(item.passed for item in evaluations):
            raise WorkflowServiceValidationError("A passing evaluation is required before publication.")
        now = self._now()
        deployment = WorkflowDeployment(
            deployment_id=_new_id("wfd"), workflow_id=workflow.workflow_id,
            revision_id=revision.revision_id, deployed_at=now, deployed_by=principal.subject,
        )
        return PublishedWorkflow(
            workflow=self._store.publish(
                principal, deployment, updated_at=now,
                expected_active_revision_id=expected_active_revision_id,
            ),
            deployment=deployment,
        )

    def decide_approval(
        self, principal: Principal, approval_id: str, decision: ApprovalDecision
    ) -> None:
        _require_operator(principal)
        self._store.decide_approval(principal, approval_id, decision=decision, decided_at=self._now())

    def _verify_resources(self, principal: Principal, specification: WorkflowSpec) -> None:
        for node in specification.nodes:
            for resource in node.resources:
                if resource.resource_kind is WorkflowResourceKind.MODEL_PROFILE:
                    if resource.resource_id not in self._trusted_model_profile_ids:
                        raise WorkflowServiceValidationError("The model profile is unavailable.")
                    continue
                try:
                    record = self._knowledge_bases.get(principal, resource.resource_id)
                except Exception as error:
                    raise WorkflowResourceUnavailableError() from error
                if record.status is not KnowledgeBaseStatus.READY:
                    raise WorkflowResourceUnavailableError()

    def _now(self) -> float:
        value = float(self._clock())
        if not is_valid_timestamp(value):
            raise WorkflowServiceValidationError("clock returned an invalid timestamp")
        return value


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(16)}"


def _require_writer(principal: Principal) -> None:
    if not isinstance(principal, Principal) or not principal.has_role("writer"):
        raise WorkflowAuthorizationError()


def _require_operator(principal: Principal) -> None:
    if not isinstance(principal, Principal) or not principal.has_role("operator"):
        raise WorkflowAuthorizationError()


__all__ = [
    "PublishedWorkflow",
    "WorkflowAuthorizationError",
    "WorkflowResourceUnavailableError",
    "WorkflowService",
    "WorkflowServiceError",
    "WorkflowServiceValidationError",
]
