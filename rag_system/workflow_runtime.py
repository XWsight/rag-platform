"""Bounded, resumable execution for immutable workflow revisions.

The runtime deliberately accepts only registered native node executors.  It
does not evaluate user code, invoke shells, or resolve credentials from a
workflow definition.  Every state transition and node result is persisted
through :mod:`workflow_store` before execution advances.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from rag_system.tenancy import Principal
from rag_system.workflow_contracts import WorkflowNode, WorkflowNodeKind, WorkflowSpec
from rag_system.workflow_models import (
    ApprovalDecision,
    WorkflowApproval,
    WorkflowRevision,
    WorkflowRun,
    WorkflowRunState,
    WorkflowRunStatus,
    WorkflowStepRun,
    WorkflowStepStatus,
)
from rag_system.workflow_store import WorkflowStore


class WorkflowRuntimeError(Exception):
    """A safe, structured workflow execution refusal or failure."""


class WorkflowNotPublishedError(WorkflowRuntimeError):
    def __init__(self) -> None:
        super().__init__("Workflow is not published.")


class WorkflowExecutionBudgetError(WorkflowRuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__("Workflow execution budget was exceeded.")
        self.code = code


class WorkflowNodeExecutionError(WorkflowRuntimeError):
    def __init__(self, code: str = "node_execution_failed") -> None:
        super().__init__("Workflow node execution failed.")
        self.code = code


class NativeWorkflowNodeExecutor(Protocol):
    """Implementation port for one pre-approved native node kind."""

    def __call__(
        self,
        principal: Principal,
        node: WorkflowNode,
        values: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class WorkflowExecution:
    run: WorkflowRun
    outputs: Mapping[str, Any] | None
    pending_approval_id: str | None = None

    def __post_init__(self) -> None:
        if self.pending_approval_id is not None and self.run.status is not WorkflowRunStatus.WAITING_APPROVAL:
            raise ValueError("only a waiting run can have a pending approval")
        if self.run.status is WorkflowRunStatus.SUCCEEDED and self.outputs is None:
            raise ValueError("a successful run must have outputs")
        if self.run.status is not WorkflowRunStatus.SUCCEEDED and self.outputs is not None:
            raise ValueError("only a successful run can have outputs")


class WorkflowRuntime:
    """Execute one active revision with hard bounds and durable pause/resume."""

    def __init__(
        self,
        store: WorkflowStore,
        executors: Mapping[WorkflowNodeKind, NativeWorkflowNodeExecutor],
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not isinstance(executors, Mapping):
            raise TypeError("executors must be a mapping")
        normalized: dict[WorkflowNodeKind, NativeWorkflowNodeExecutor] = {}
        for kind, executor in executors.items():
            if not isinstance(kind, WorkflowNodeKind) or not callable(executor):
                raise TypeError("workflow executors must have native node kinds")
            normalized[kind] = executor
        self._store = store
        self._executors = normalized
        self._clock = clock

    def start(
        self, principal: Principal, workflow_id: str, input_values: Mapping[str, Any]
    ) -> WorkflowExecution:
        workflow = self._store.get_workflow(principal, workflow_id)
        if workflow.active_revision_id is None:
            raise WorkflowNotPublishedError()
        revision = self._store.get_revision(principal, workflow.workflow_id, workflow.active_revision_id)
        inputs = _validate_inputs(revision.specification, input_values)
        now = self._now()
        run = WorkflowRun(
            run_id=_new_id("wrun"),
            workflow_id=workflow.workflow_id,
            revision_id=revision.revision_id,
            specification_digest=revision.specification_digest,
            status=WorkflowRunStatus.CREATED,
            created_at=now,
            updated_at=now,
            created_by=principal.subject,
            input_digest=_digest(inputs),
        )
        created = self._store.create_run(principal, run)
        self._store.save_run_state(
            principal,
            WorkflowRunState(
                run_id=created.run_id, input_values=inputs, node_outputs={}, updated_at=now
            ),
        )
        queued = self._store.transition_run(
            principal, created.run_id, status=WorkflowRunStatus.QUEUED, updated_at=now
        )
        return self._run(principal, revision, queued)

    def resume(self, principal: Principal, run_id: str) -> WorkflowExecution:
        run = self._store.get_run(principal, run_id)
        if run.status is not WorkflowRunStatus.WAITING_APPROVAL:
            raise WorkflowRuntimeError("Workflow run is not awaiting approval.")
        approvals = self._store.list_approvals(principal, run.run_id)
        pending = next((item for item in approvals if item.decision is None), None)
        if pending is not None:
            return WorkflowExecution(run=run, outputs=None, pending_approval_id=pending.approval_id)
        revision = self._store.get_revision(principal, run.workflow_id, run.revision_id)
        now = self._now()
        queued = self._store.transition_run(
            principal, run.run_id, status=WorkflowRunStatus.QUEUED, updated_at=now
        )
        return self._run(principal, revision, queued)

    def _run(
        self, principal: Principal, revision: WorkflowRevision, run: WorkflowRun
    ) -> WorkflowExecution:
        now = self._now()
        running = self._store.transition_run(
            principal, run.run_id, status=WorkflowRunStatus.RUNNING, updated_at=now
        )
        state = self._store.get_run_state(principal, running.run_id)
        node_outputs = {key: dict(value) for key, value in state.node_outputs.items()}
        try:
            for node in _execution_order(revision.specification):
                if node.node_id in node_outputs:
                    continue
                self._enforce_budget(revision, running, node_outputs)
                node_inputs = _resolve_node_inputs(node, state.input_values, node_outputs)
                if node.node_kind is WorkflowNodeKind.HUMAN_APPROVAL:
                    approval = self._approval_for_node(principal, running.run_id, node.node_id)
                    if approval is None:
                        now = self._now()
                        created = self._store.create_approval(
                            principal,
                            WorkflowApproval(
                                approval_id=_new_id("wappr"), run_id=running.run_id,
                                node_id=node.node_id, requested_at=now, requested_by=principal.subject,
                            ),
                        )
                        self._save_step(principal, running.run_id, node, node_inputs,
                                        WorkflowStepStatus.WAITING_APPROVAL, now, None, None)
                        waiting = self._store.transition_run(
                            principal, running.run_id, status=WorkflowRunStatus.WAITING_APPROVAL,
                            updated_at=now,
                        )
                        self._save_state(principal, waiting.run_id, state.input_values, node_outputs, now)
                        return WorkflowExecution(waiting, None, created.approval_id)
                    if approval.decision is None:
                        waiting = self._store.transition_run(
                            principal, running.run_id, status=WorkflowRunStatus.WAITING_APPROVAL,
                            updated_at=self._now(),
                        )
                        return WorkflowExecution(waiting, None, approval.approval_id)
                    if approval.decision is ApprovalDecision.REJECTED:
                        self._save_step(principal, running.run_id, node, node_inputs,
                                        WorkflowStepStatus.FAILED, self._now(), self._now(), None,
                                        error_code="approval_rejected")
                        failed = self._store.transition_run(
                            principal, running.run_id, status=WorkflowRunStatus.FAILED,
                            updated_at=self._now(), error_code="approval_rejected",
                        )
                        return WorkflowExecution(failed, None)
                    output = {"decision": ApprovalDecision.APPROVED.value}
                else:
                    output = self._execute_node(principal, node, node_inputs)
                node_outputs[node.node_id] = output
                now = self._now()
                self._save_step(principal, running.run_id, node, node_inputs,
                                WorkflowStepStatus.SUCCEEDED, now, now, output)
                self._save_state(principal, running.run_id, state.input_values, node_outputs, now)
            outputs = _public_outputs(revision.specification, node_outputs)
            succeeded = self._store.transition_run(
                principal, running.run_id, status=WorkflowRunStatus.SUCCEEDED, updated_at=self._now()
            )
            return WorkflowExecution(succeeded, outputs)
        except WorkflowExecutionBudgetError as error:
            failed = self._store.transition_run(
                principal, running.run_id, status=WorkflowRunStatus.FAILED,
                updated_at=self._now(), error_code=error.code,
            )
            return WorkflowExecution(failed, None)
        except WorkflowRuntimeError as error:
            failed = self._store.transition_run(
                principal, running.run_id, status=WorkflowRunStatus.FAILED,
                updated_at=self._now(), error_code=getattr(error, "code", "node_execution_failed"),
            )
            return WorkflowExecution(failed, None)

    def _execute_node(
        self, principal: Principal, node: WorkflowNode, values: Mapping[str, Any]
    ) -> dict[str, Any]:
        executor = self._executors.get(node.node_kind)
        if executor is None:
            raise WorkflowNodeExecutionError("node_executor_unavailable")
        try:
            output = executor(principal, node, values)
        except WorkflowRuntimeError:
            raise
        except Exception as error:
            raise WorkflowNodeExecutionError() from error
        if not isinstance(output, Mapping) or set(output) != set(node.output_names):
            raise WorkflowNodeExecutionError("node_output_invalid")
        try:
            encoded = json.dumps(output, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            decoded = json.loads(encoded)
        except (TypeError, ValueError) as error:
            raise WorkflowNodeExecutionError("node_output_invalid") from error
        if not isinstance(decoded, dict) or len(encoded.encode("utf-8")) > 256 * 1024:
            raise WorkflowNodeExecutionError("node_output_invalid")
        return decoded

    def _enforce_budget(
        self, revision: WorkflowRevision, run: WorkflowRun, outputs: Mapping[str, Mapping[str, Any]]
    ) -> None:
        if len(outputs) >= revision.budget.max_steps:
            raise WorkflowExecutionBudgetError("step_budget_exceeded")
        if self._now() - run.created_at > revision.budget.max_wall_seconds:
            raise WorkflowExecutionBudgetError("wall_time_budget_exceeded")
        model_nodes = {item.node_id for item in revision.specification.nodes if item.node_kind is WorkflowNodeKind.MODEL_GENERATE}
        if len(model_nodes & set(outputs)) >= revision.budget.max_model_calls:
            raise WorkflowExecutionBudgetError("model_call_budget_exceeded")

    def _approval_for_node(
        self, principal: Principal, run_id: str, node_id: str
    ) -> WorkflowApproval | None:
        items = [item for item in self._store.list_approvals(principal, run_id) if item.node_id == node_id]
        if len(items) > 1:
            raise WorkflowNodeExecutionError("approval_state_invalid")
        return items[0] if items else None

    def _save_step(
        self, principal: Principal, run_id: str, node: WorkflowNode, values: Mapping[str, Any],
        status: WorkflowStepStatus, started_at: float, finished_at: float | None,
        output: Mapping[str, Any] | None, *, error_code: str | None = None,
    ) -> None:
        self._store.save_step_run(
            principal,
            WorkflowStepRun(
                step_run_id=_step_id(run_id, node.node_id), run_id=run_id, node_id=node.node_id,
                status=status, started_at=started_at, finished_at=finished_at,
                input_digest=_digest(values), output_digest=_digest(output) if output is not None else None,
                error_code=error_code,
            ),
        )

    def _save_state(self, principal: Principal, run_id: str, inputs: Mapping[str, Any],
                    outputs: Mapping[str, Mapping[str, Any]], updated_at: float) -> None:
        self._store.save_run_state(
            principal,
            WorkflowRunState(run_id=run_id, input_values=inputs, node_outputs=outputs, updated_at=updated_at),
        )

    def _now(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise RuntimeError("clock returned an invalid timestamp")
        return float(value)


def _execution_order(specification: WorkflowSpec) -> tuple[WorkflowNode, ...]:
    nodes = {node.node_id: node for node in specification.nodes}
    remaining = {node_id: set(node.depends_on) for node_id, node in nodes.items()}
    ordered: list[WorkflowNode] = []
    while remaining:
        ready = sorted(node_id for node_id, dependencies in remaining.items() if not dependencies)
        if not ready:
            raise WorkflowRuntimeError("Workflow graph is invalid.")
        for node_id in ready:
            ordered.append(nodes[node_id])
            del remaining[node_id]
        completed = set(ready)
        for dependencies in remaining.values():
            dependencies.difference_update(completed)
    return tuple(ordered)


def _validate_inputs(specification: WorkflowSpec, values: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise WorkflowRuntimeError("Workflow inputs must be an object.")
    expected = {item.name: item for item in specification.inputs}
    if not set(values) <= set(expected):
        raise WorkflowRuntimeError("Workflow inputs contain an unknown field.")
    if any(item.required and item.name not in values for item in expected.values()):
        raise WorkflowRuntimeError("Workflow inputs are incomplete.")
    try:
        encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise WorkflowRuntimeError("Workflow inputs must be JSON-compatible.") from error
    if not isinstance(decoded, dict) or len(encoded.encode("utf-8")) > 256 * 1024:
        raise WorkflowRuntimeError("Workflow inputs are too large.")
    return decoded


def _resolve_node_inputs(node: WorkflowNode, inputs: Mapping[str, Any],
                         outputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for binding in node.input_bindings:
        parts = binding.source.split(".")
        if parts[0] == "input":
            resolved[binding.target] = inputs[parts[1]]
        else:
            resolved[binding.target] = outputs[parts[1]][parts[2]]
    return resolved


def _public_outputs(specification: WorkflowSpec, outputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {output.name: outputs[output.source.split(".")[1]][output.source.split(".")[2]] for output in specification.outputs}


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(16)}"


def _step_id(run_id: str, node_id: str) -> str:
    """Stable per-run ID lets resume update the same node audit record."""

    return "wstep_" + hashlib.sha256(f"{run_id}:{node_id}".encode()).hexdigest()[:32]


__all__ = [
    "NativeWorkflowNodeExecutor",
    "WorkflowExecution",
    "WorkflowExecutionBudgetError",
    "WorkflowNodeExecutionError",
    "WorkflowNotPublishedError",
    "WorkflowRuntime",
    "WorkflowRuntimeError",
]
