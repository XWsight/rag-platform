"""Storage-neutral identities and durable state contracts for workflows."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, cast

from rag_system.application_contracts import (
    is_valid_timestamp,
    validate_change_summary,
    validate_display_name,
    validate_project_id,
    validate_subject,
)
from rag_system.tenancy import TenantId
from rag_system.workflow_contracts import WorkflowSpec


_WORKFLOW_ID = re.compile(r"wf_[A-Za-z0-9_-]{32}")
_WORKFLOW_REVISION_ID = re.compile(r"wfr_[A-Za-z0-9_-]{32}")
_WORKFLOW_DEPLOYMENT_ID = re.compile(r"wfd_[A-Za-z0-9_-]{32}")
_WORKFLOW_RUN_ID = re.compile(r"wrun_[A-Za-z0-9_-]{32}")
_WORKFLOW_STEP_RUN_ID = re.compile(r"wstep_[A-Za-z0-9_-]{32}")
_WORKFLOW_APPROVAL_ID = re.compile(r"wappr_[A-Za-z0-9_-]{32}")
_WORKFLOW_EVALUATION_ID = re.compile(r"weval_[A-Za-z0-9_-]{32}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


class WorkflowModelError(ValueError):
    """A durable workflow value violates the platform contract."""


class WorkflowStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class WorkflowDeploymentStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"


class WorkflowRunStatus(StrEnum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class WorkflowStepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ExecutionBudget:
    """Revision-scoped bounds that every run must obey before executing nodes."""

    max_steps: int = 32
    max_model_calls: int = 4
    max_wall_seconds: int = 120

    def __post_init__(self) -> None:
        _validate_int(self.max_steps, "max_steps", minimum=1, maximum=64)
        _validate_int(self.max_model_calls, "max_model_calls", minimum=0, maximum=32)
        _validate_int(self.max_wall_seconds, "max_wall_seconds", minimum=1, maximum=3_600)


@dataclass(frozen=True, slots=True)
class Workflow:
    workflow_id: str
    tenant_id: TenantId
    project_id: str
    display_name: str
    active_revision_id: str | None
    status: WorkflowStatus
    created_at: float
    updated_at: float

    def __post_init__(self) -> None:
        validate_workflow_id(self.workflow_id)
        if not isinstance(self.tenant_id, TenantId):
            raise WorkflowModelError("workflow tenant is invalid")
        validate_project_id(self.project_id)
        object.__setattr__(self, "display_name", validate_display_name(self.display_name))
        if self.active_revision_id is not None:
            validate_workflow_revision_id(self.active_revision_id)
        if not isinstance(self.status, WorkflowStatus):
            raise WorkflowModelError("workflow status is invalid")
        _validate_time_range(self.created_at, self.updated_at)


@dataclass(frozen=True, slots=True)
class WorkflowDraft:
    workflow_id: str
    version: int
    specification: WorkflowSpec | None
    budget: ExecutionBudget | None
    updated_at: float
    updated_by: str
    change_summary: str = ""

    def __post_init__(self) -> None:
        validate_workflow_id(self.workflow_id)
        _validate_int(self.version, "draft version", minimum=0, maximum=2_147_483_647)
        if self.specification is not None and not isinstance(self.specification, WorkflowSpec):
            raise WorkflowModelError("workflow draft specification is invalid")
        if self.budget is not None and not isinstance(self.budget, ExecutionBudget):
            raise WorkflowModelError("workflow draft budget is invalid")
        if (self.specification is None) != (self.budget is None):
            raise WorkflowModelError("workflow draft specification and budget must be set together")
        if not is_valid_timestamp(self.updated_at):
            raise WorkflowModelError("workflow draft timestamp is invalid")
        object.__setattr__(self, "updated_by", validate_subject(self.updated_by))
        if self.specification is None:
            if self.change_summary:
                raise WorkflowModelError("an empty workflow draft cannot have a change summary")
        else:
            object.__setattr__(self, "change_summary", validate_change_summary(self.change_summary))


@dataclass(frozen=True, slots=True)
class WorkflowRevision:
    revision_id: str
    workflow_id: str
    revision_number: int
    specification: WorkflowSpec
    budget: ExecutionBudget
    created_at: float
    created_by: str
    change_summary: str

    def __post_init__(self) -> None:
        validate_workflow_revision_id(self.revision_id)
        validate_workflow_id(self.workflow_id)
        _validate_int(self.revision_number, "workflow revision number", minimum=1, maximum=2_147_483_647)
        if not isinstance(self.specification, WorkflowSpec):
            raise WorkflowModelError("workflow revision specification is invalid")
        if not isinstance(self.budget, ExecutionBudget):
            raise WorkflowModelError("workflow revision budget is invalid")
        if not is_valid_timestamp(self.created_at):
            raise WorkflowModelError("workflow revision timestamp is invalid")
        object.__setattr__(self, "created_by", validate_subject(self.created_by))
        object.__setattr__(self, "change_summary", validate_change_summary(self.change_summary))

    @property
    def specification_digest(self) -> str:
        return self.specification.digest


@dataclass(frozen=True, slots=True)
class WorkflowDeployment:
    deployment_id: str
    workflow_id: str
    revision_id: str
    deployed_at: float
    deployed_by: str
    status: WorkflowDeploymentStatus = WorkflowDeploymentStatus.ACTIVE

    def __post_init__(self) -> None:
        validate_workflow_deployment_id(self.deployment_id)
        validate_workflow_id(self.workflow_id)
        validate_workflow_revision_id(self.revision_id)
        if not is_valid_timestamp(self.deployed_at):
            raise WorkflowModelError("workflow deployment timestamp is invalid")
        object.__setattr__(self, "deployed_by", validate_subject(self.deployed_by))
        if not isinstance(self.status, WorkflowDeploymentStatus):
            raise WorkflowModelError("workflow deployment status is invalid")


@dataclass(frozen=True, slots=True)
class WorkflowRun:
    run_id: str
    workflow_id: str
    revision_id: str
    specification_digest: str
    status: WorkflowRunStatus
    created_at: float
    updated_at: float
    created_by: str
    input_digest: str
    error_code: str | None = None

    def __post_init__(self) -> None:
        validate_workflow_run_id(self.run_id)
        validate_workflow_id(self.workflow_id)
        validate_workflow_revision_id(self.revision_id)
        _validate_digest(self.specification_digest, "workflow specification digest")
        _validate_digest(self.input_digest, "workflow input digest")
        if not isinstance(self.status, WorkflowRunStatus):
            raise WorkflowModelError("workflow run status is invalid")
        _validate_time_range(self.created_at, self.updated_at)
        object.__setattr__(self, "created_by", validate_subject(self.created_by))
        if self.error_code is not None:
            object.__setattr__(self, "error_code", _validate_error_code(self.error_code))


@dataclass(frozen=True, slots=True)
class WorkflowStepRun:
    step_run_id: str
    run_id: str
    node_id: str
    status: WorkflowStepStatus
    started_at: float | None
    finished_at: float | None
    input_digest: str | None
    output_digest: str | None
    error_code: str | None = None

    def __post_init__(self) -> None:
        validate_workflow_step_run_id(self.step_run_id)
        validate_workflow_run_id(self.run_id)
        if not isinstance(self.node_id, str) or not self.node_id:
            raise WorkflowModelError("workflow step node ID is invalid")
        if not isinstance(self.status, WorkflowStepStatus):
            raise WorkflowModelError("workflow step status is invalid")
        _validate_optional_timestamp(self.started_at, "workflow step start timestamp")
        _validate_optional_timestamp(self.finished_at, "workflow step finish timestamp")
        if self.started_at is None and self.finished_at is not None:
            raise WorkflowModelError("workflow step cannot finish before it starts")
        if self.started_at is not None and self.finished_at is not None and self.finished_at < self.started_at:
            raise WorkflowModelError("workflow step finish timestamp is invalid")
        _validate_optional_digest(self.input_digest, "workflow step input digest")
        _validate_optional_digest(self.output_digest, "workflow step output digest")
        if self.error_code is not None:
            object.__setattr__(self, "error_code", _validate_error_code(self.error_code))


@dataclass(frozen=True, slots=True)
class WorkflowApproval:
    approval_id: str
    run_id: str
    node_id: str
    requested_at: float
    requested_by: str
    decision: ApprovalDecision | None = None
    decided_at: float | None = None
    decided_by: str | None = None

    def __post_init__(self) -> None:
        validate_workflow_approval_id(self.approval_id)
        validate_workflow_run_id(self.run_id)
        if not isinstance(self.node_id, str) or not self.node_id:
            raise WorkflowModelError("workflow approval node ID is invalid")
        if not is_valid_timestamp(self.requested_at):
            raise WorkflowModelError("workflow approval timestamp is invalid")
        object.__setattr__(self, "requested_by", validate_subject(self.requested_by))
        if self.decision is None:
            if self.decided_at is not None or self.decided_by is not None:
                raise WorkflowModelError("pending workflow approval cannot be decided")
            return
        if not isinstance(self.decision, ApprovalDecision):
            raise WorkflowModelError("workflow approval decision is invalid")
        if not is_valid_timestamp(self.decided_at) or _timestamp(self.decided_at) < _timestamp(
            self.requested_at
        ):
            raise WorkflowModelError("workflow approval decision timestamp is invalid")
        object.__setattr__(self, "decided_by", validate_subject(self.decided_by))


@dataclass(frozen=True, slots=True)
class WorkflowRunState:
    """Tenant-protected resumable state for a paused workflow run.

    The workflow definition never contains secrets.  Runtime values are stored
    only while a run is active or awaiting approval, and are deliberately
    bounded to keep a single run from exhausting the local durable profile.
    """

    run_id: str
    input_values: Mapping[str, Any]
    node_outputs: Mapping[str, Mapping[str, Any]]
    updated_at: float

    def __post_init__(self) -> None:
        validate_workflow_run_id(self.run_id)
        if not is_valid_timestamp(self.updated_at):
            raise WorkflowModelError("workflow run state timestamp is invalid")
        encoded_inputs = _canonical_json_object(self.input_values, "workflow run inputs")
        normalized_inputs = json.loads(encoded_inputs)
        normalized_outputs: dict[str, Mapping[str, Any]] = {}
        if not isinstance(self.node_outputs, Mapping):
            raise WorkflowModelError("workflow run outputs are invalid")
        for node_id, output in self.node_outputs.items():
            if not isinstance(node_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", node_id):
                raise WorkflowModelError("workflow run output node ID is invalid")
            encoded_output = _canonical_json_object(output, "workflow node output")
            normalized_outputs[node_id] = MappingProxyType(json.loads(encoded_output))
        encoded_outputs = json.dumps(
            {key: dict(value) for key, value in sorted(normalized_outputs.items())},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len((encoded_inputs + encoded_outputs).encode("utf-8")) > 512 * 1024:
            raise WorkflowModelError("workflow run state is too large")
        object.__setattr__(self, "input_values", MappingProxyType(normalized_inputs))
        object.__setattr__(self, "node_outputs", MappingProxyType(normalized_outputs))

    @property
    def input_json(self) -> str:
        return json.dumps(dict(self.input_values), ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @property
    def outputs_json(self) -> str:
        return json.dumps(
            {key: dict(value) for key, value in self.node_outputs.items()},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class WorkflowEvaluation:
    """Immutable release evidence tied to the exact workflow specification."""

    evaluation_id: str
    workflow_id: str
    revision_id: str
    specification_digest: str
    generated_at: float
    case_count: int
    passed_case_count: int

    def __post_init__(self) -> None:
        validate_workflow_evaluation_id(self.evaluation_id)
        validate_workflow_id(self.workflow_id)
        validate_workflow_revision_id(self.revision_id)
        _validate_digest(self.specification_digest, "workflow evaluation specification digest")
        if not is_valid_timestamp(self.generated_at):
            raise WorkflowModelError("workflow evaluation timestamp is invalid")
        _validate_int(self.case_count, "workflow evaluation case count", minimum=1, maximum=100_000)
        _validate_int(
            self.passed_case_count, "workflow evaluation passing case count",
            minimum=0, maximum=self.case_count,
        )

    @property
    def passed(self) -> bool:
        """Production gating is intentionally strict for the first runtime profile."""

        return self.passed_case_count == self.case_count


def validate_workflow_id(value: object) -> str:
    return _validate_id(value, _WORKFLOW_ID, "workflow ID")


def validate_workflow_revision_id(value: object) -> str:
    return _validate_id(value, _WORKFLOW_REVISION_ID, "workflow revision ID")


def validate_workflow_deployment_id(value: object) -> str:
    return _validate_id(value, _WORKFLOW_DEPLOYMENT_ID, "workflow deployment ID")


def validate_workflow_run_id(value: object) -> str:
    return _validate_id(value, _WORKFLOW_RUN_ID, "workflow run ID")


def validate_workflow_step_run_id(value: object) -> str:
    return _validate_id(value, _WORKFLOW_STEP_RUN_ID, "workflow step run ID")


def validate_workflow_approval_id(value: object) -> str:
    return _validate_id(value, _WORKFLOW_APPROVAL_ID, "workflow approval ID")


def validate_workflow_evaluation_id(value: object) -> str:
    return _validate_id(value, _WORKFLOW_EVALUATION_ID, "workflow evaluation ID")


def _validate_id(value: object, pattern: re.Pattern[str], description: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise WorkflowModelError(f"{description} has an invalid format")
    return value


def _validate_int(value: object, description: str, *, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise WorkflowModelError(f"{description} is invalid")


def _validate_time_range(created_at: object, updated_at: object) -> None:
    if not is_valid_timestamp(created_at) or not is_valid_timestamp(updated_at):
        raise WorkflowModelError("workflow timestamps are invalid")
    if _timestamp(updated_at) < _timestamp(created_at):
        raise WorkflowModelError("workflow timestamps are invalid")


def _validate_optional_timestamp(value: object, description: str) -> None:
    if value is not None and not is_valid_timestamp(value):
        raise WorkflowModelError(f"{description} is invalid")


def _timestamp(value: object) -> float:
    return float(cast(int | float, value))


def _validate_digest(value: object, description: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise WorkflowModelError(f"{description} is invalid")


def _validate_optional_digest(value: object, description: str) -> None:
    if value is not None:
        _validate_digest(value, description)


def _validate_error_code(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
        raise WorkflowModelError("workflow error code is invalid")
    return value


def _canonical_json_object(value: object, description: str) -> str:
    if not isinstance(value, Mapping):
        raise WorkflowModelError(f"{description} are invalid")
    try:
        encoded = json.dumps(dict(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise WorkflowModelError(f"{description} must be JSON-compatible") from error
    if not isinstance(decoded, dict):
        raise WorkflowModelError(f"{description} are invalid")
    return encoded


__all__ = [
    "ApprovalDecision",
    "ExecutionBudget",
    "Workflow",
    "WorkflowApproval",
    "WorkflowDeployment",
    "WorkflowDeploymentStatus",
    "WorkflowDraft",
    "WorkflowEvaluation",
    "WorkflowModelError",
    "WorkflowRevision",
    "WorkflowRun",
    "WorkflowRunState",
    "WorkflowRunStatus",
    "WorkflowStatus",
    "WorkflowStepRun",
    "WorkflowStepStatus",
    "validate_workflow_approval_id",
    "validate_workflow_deployment_id",
    "validate_workflow_evaluation_id",
    "validate_workflow_id",
    "validate_workflow_revision_id",
    "validate_workflow_run_id",
    "validate_workflow_step_run_id",
]
