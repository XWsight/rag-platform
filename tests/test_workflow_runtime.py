from __future__ import annotations

import secrets
import tempfile
import unittest
from pathlib import Path

from rag_system.tenancy import Principal, TenantId
from rag_system.workflow_contracts import (
    WORKFLOW_DSL_SCHEMA_VERSION,
    WorkflowInput,
    WorkflowInputBinding,
    WorkflowNode,
    WorkflowNodeKind,
    WorkflowOutput,
    WorkflowResourceKind,
    WorkflowResourceRef,
    WorkflowSpec,
)
from rag_system.workflow_models import (
    ApprovalDecision,
    ExecutionBudget,
    Workflow,
    WorkflowDeployment,
    WorkflowDraft,
    WorkflowRevision,
    WorkflowRunStatus,
    WorkflowStatus,
)
from rag_system.workflow_runtime import WorkflowRuntime
from rag_system.workflow_store import WorkflowStore


PROJECT_ID = "prj_12345678901234567890123456789012"
KNOWLEDGE_BASE_ID = "kb_12345678901234567890123456789012"


def _id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(16)}"


def _specification(*, approval: bool = False) -> WorkflowSpec:
    retrieve = WorkflowNode(
        node_id="retrieve", node_kind=WorkflowNodeKind.KNOWLEDGE_RETRIEVE,
        input_bindings=(WorkflowInputBinding("query", "input.question"),),
        output_names=("evidence",),
        resources=(WorkflowResourceRef(WorkflowResourceKind.KNOWLEDGE_BASE, KNOWLEDGE_BASE_ID),),
    )
    prompt = WorkflowNode(
        node_id="prompt", node_kind=WorkflowNodeKind.PROMPT_RENDER, depends_on=("retrieve",),
        input_bindings=(
            WorkflowInputBinding("question", "input.question"),
            WorkflowInputBinding("evidence", "node.retrieve.evidence"),
        ), output_names=("prompt",), parameters={"template": "{{ question }} {{ evidence }}"},
    )
    generate = WorkflowNode(
        node_id="generate", node_kind=WorkflowNodeKind.MODEL_GENERATE, depends_on=("prompt",),
        input_bindings=(WorkflowInputBinding("prompt", "node.prompt.prompt"),),
        output_names=("answer",),
        resources=(WorkflowResourceRef(WorkflowResourceKind.MODEL_PROFILE, "default"),),
    )
    nodes = [retrieve, prompt, generate]
    outputs = [WorkflowOutput("answer", "node.generate.answer")]
    if approval:
        review = WorkflowNode(
            node_id="review", node_kind=WorkflowNodeKind.HUMAN_APPROVAL, depends_on=("generate",),
            input_bindings=(WorkflowInputBinding("message", "node.generate.answer"),),
            output_names=("decision",), parameters={"timeout_seconds": 60},
        )
        nodes.append(review)
        outputs.append(WorkflowOutput("decision", "node.review.decision"))
    return WorkflowSpec(
        schema_version=WORKFLOW_DSL_SCHEMA_VERSION, inputs=(WorkflowInput("question"),),
        nodes=tuple(nodes), outputs=tuple(outputs),
    )


def _executors():
    return {
        WorkflowNodeKind.KNOWLEDGE_RETRIEVE: lambda _p, _n, values: {"evidence": f"source:{values['query']}"},
        WorkflowNodeKind.PROMPT_RENDER: lambda _p, _n, values: {"prompt": f"{values['question']}|{values['evidence']}"},
        WorkflowNodeKind.MODEL_GENERATE: lambda _p, _n, values: {"answer": f"answer:{values['prompt']}"},
    }


class WorkflowRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = WorkflowStore(Path(self.tempdir.name) / "workflows.sqlite3")
        self.principal = Principal("operator", TenantId("tenant-a"), frozenset({"reader", "writer", "operator"}))
        self.now = 100.0
        self.runtime = WorkflowRuntime(self.store, _executors(), clock=lambda: self.now)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _publish(self, specification: WorkflowSpec, budget: ExecutionBudget | None = None) -> Workflow:
        budget = budget or ExecutionBudget()
        workflow = self.store.create_workflow(
            self.principal,
            Workflow(_id("wf"), self.principal.tenant_id, PROJECT_ID, "Workflow", None,
                     WorkflowStatus.ACTIVE, self.now, self.now),
        )
        draft = WorkflowDraft(workflow.workflow_id, 1, specification, budget, self.now, self.principal.subject, "Initial")
        self.store.update_draft(self.principal, draft, expected_version=0)
        revision = self.store.create_revision(
            self.principal,
            WorkflowRevision(_id("wfr"), workflow.workflow_id, 1, specification, budget,
                             self.now, self.principal.subject, "Initial"),
        )
        return self.store.publish(
            self.principal,
            WorkflowDeployment(_id("wfd"), workflow.workflow_id, revision.revision_id, self.now, self.principal.subject),
            updated_at=self.now, expected_active_revision_id=None,
        )

    def test_executes_native_nodes_and_persists_auditable_steps(self) -> None:
        workflow = self._publish(_specification())

        execution = self.runtime.start(self.principal, workflow.workflow_id, {"question": "hello"})

        self.assertEqual(execution.run.status, WorkflowRunStatus.SUCCEEDED)
        self.assertEqual(execution.outputs, {"answer": "answer:hello|source:hello"})
        self.assertEqual(
            [step.status.value for step in self.store.list_step_runs(self.principal, execution.run.run_id)],
            ["succeeded", "succeeded", "succeeded"],
        )
        state = self.store.get_run_state(self.principal, execution.run.run_id)
        self.assertIn("generate", state.node_outputs)

    def test_approval_pauses_then_resumes_from_durable_state(self) -> None:
        workflow = self._publish(_specification(approval=True))
        waiting = self.runtime.start(self.principal, workflow.workflow_id, {"question": "hello"})

        self.assertEqual(waiting.run.status, WorkflowRunStatus.WAITING_APPROVAL)
        self.assertIsNotNone(waiting.pending_approval_id)
        self.store.decide_approval(
            self.principal, waiting.pending_approval_id or "", decision=ApprovalDecision.APPROVED,
            decided_at=self.now + 1,
        )
        self.now += 1
        execution = self.runtime.resume(self.principal, waiting.run.run_id)

        self.assertEqual(execution.run.status, WorkflowRunStatus.SUCCEEDED)
        self.assertEqual(execution.outputs["decision"] if execution.outputs else None, "approved")
        self.assertEqual(len(self.store.list_step_runs(self.principal, waiting.run.run_id)), 4)

    def test_budget_failure_stops_before_disallowed_model_call(self) -> None:
        workflow = self._publish(_specification(), ExecutionBudget(max_steps=10, max_model_calls=0, max_wall_seconds=60))

        execution = self.runtime.start(self.principal, workflow.workflow_id, {"question": "hello"})

        self.assertEqual(execution.run.status, WorkflowRunStatus.FAILED)
        self.assertEqual(execution.run.error_code, "model_call_budget_exceeded")


if __name__ == "__main__":
    unittest.main()
