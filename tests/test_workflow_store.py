from __future__ import annotations

import hashlib
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
    WorkflowApproval,
    WorkflowDeployment,
    WorkflowRevision,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowStatus,
    WorkflowStepRun,
    WorkflowStepStatus,
)
from rag_system.workflow_store import (
    WorkflowDraftConflictError,
    WorkflowStore,
    WorkflowUnavailableError,
)


PROJECT_ID = "prj_12345678901234567890123456789012"
KNOWLEDGE_BASE_ID = "kb_12345678901234567890123456789012"


def _id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(16)}"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _specification() -> WorkflowSpec:
    retrieve = WorkflowNode(
        node_id="retrieve",
        node_kind=WorkflowNodeKind.KNOWLEDGE_RETRIEVE,
        input_bindings=(WorkflowInputBinding("query", "input.question"),),
        output_names=("evidence",),
        resources=(WorkflowResourceRef(WorkflowResourceKind.KNOWLEDGE_BASE, KNOWLEDGE_BASE_ID),),
    )
    prompt = WorkflowNode(
        node_id="prompt",
        node_kind=WorkflowNodeKind.PROMPT_RENDER,
        depends_on=("retrieve",),
        input_bindings=(
            WorkflowInputBinding("question", "input.question"),
            WorkflowInputBinding("evidence", "node.retrieve.evidence"),
        ),
        output_names=("prompt",),
        parameters={"template": "{{ question }}\n{{ evidence }}"},
    )
    generate = WorkflowNode(
        node_id="generate",
        node_kind=WorkflowNodeKind.MODEL_GENERATE,
        depends_on=("prompt",),
        input_bindings=(WorkflowInputBinding("prompt", "node.prompt.prompt"),),
        output_names=("answer",),
        resources=(WorkflowResourceRef(WorkflowResourceKind.MODEL_PROFILE, "default"),),
    )
    return WorkflowSpec(
        schema_version=WORKFLOW_DSL_SCHEMA_VERSION,
        inputs=(WorkflowInput("question"),),
        nodes=(retrieve, prompt, generate),
        outputs=(WorkflowOutput("answer", "node.generate.answer"),),
    )


class WorkflowStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = WorkflowStore(Path(self.tempdir.name) / "workflows.sqlite3")
        self.principal = Principal("writer", TenantId("tenant-a"), frozenset({"reader", "writer"}))
        self.other = Principal("reader", TenantId("tenant-b"), frozenset({"reader"}))
        self.workflow = Workflow(
            workflow_id=_id("wf"),
            tenant_id=self.principal.tenant_id,
            project_id=PROJECT_ID,
            display_name="Trusted answer",
            active_revision_id=None,
            status=WorkflowStatus.ACTIVE,
            created_at=1.0,
            updated_at=1.0,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_draft_revision_and_publish_are_durable_and_tenant_scoped(self) -> None:
        self.store.create_workflow(self.principal, self.workflow)
        draft = self.store.get_draft(self.principal, self.workflow.workflow_id)
        configured = type(draft)(
            workflow_id=draft.workflow_id,
            version=1,
            specification=_specification(),
            budget=ExecutionBudget(max_steps=10, max_model_calls=2, max_wall_seconds=60),
            updated_at=2.0,
            updated_by=self.principal.subject,
            change_summary="Initial trusted workflow",
        )
        self.store.update_draft(self.principal, configured, expected_version=0)
        revision = self.store.create_revision(
            self.principal,
            WorkflowRevision(
                revision_id=_id("wfr"),
                workflow_id=self.workflow.workflow_id,
                revision_number=1,
                specification=configured.specification,
                budget=configured.budget,
                created_at=3.0,
                created_by=self.principal.subject,
                change_summary="Initial trusted workflow",
            ),
        )
        active = self.store.publish(
            self.principal,
            WorkflowDeployment(
                deployment_id=_id("wfd"),
                workflow_id=self.workflow.workflow_id,
                revision_id=revision.revision_id,
                deployed_at=4.0,
                deployed_by=self.principal.subject,
            ),
            updated_at=4.0,
            expected_active_revision_id=None,
        )

        self.assertEqual(active.active_revision_id, revision.revision_id)
        self.assertEqual(self.store.get_revision(self.principal, active.workflow_id, revision.revision_id), revision)
        with self.assertRaises(WorkflowUnavailableError):
            self.store.get_workflow(self.other, active.workflow_id)
        with self.assertRaises(WorkflowDraftConflictError):
            self.store.update_draft(self.principal, configured, expected_version=0)

    def test_run_steps_approval_and_interruption_recovery(self) -> None:
        self.store.create_workflow(self.principal, self.workflow)
        specification = _specification()
        draft = self.store.get_draft(self.principal, self.workflow.workflow_id)
        configured = type(draft)(
            workflow_id=draft.workflow_id,
            version=1,
            specification=specification,
            budget=ExecutionBudget(),
            updated_at=2.0,
            updated_by=self.principal.subject,
            change_summary="Initial workflow",
        )
        self.store.update_draft(self.principal, configured, expected_version=0)
        revision = self.store.create_revision(
            self.principal,
            WorkflowRevision(
                revision_id=_id("wfr"),
                workflow_id=self.workflow.workflow_id,
                revision_number=1,
                specification=specification,
                budget=ExecutionBudget(),
                created_at=3.0,
                created_by=self.principal.subject,
                change_summary="Initial workflow",
            ),
        )
        run = self.store.create_run(
            self.principal,
            WorkflowRun(
                run_id=_id("wrun"),
                workflow_id=self.workflow.workflow_id,
                revision_id=revision.revision_id,
                specification_digest=revision.specification_digest,
                status=WorkflowRunStatus.CREATED,
                created_at=4.0,
                updated_at=4.0,
                created_by=self.principal.subject,
                input_digest=_digest("question"),
            ),
        )
        self.store.transition_run(self.principal, run.run_id, status=WorkflowRunStatus.QUEUED, updated_at=5.0)
        running = self.store.transition_run(
            self.principal, run.run_id, status=WorkflowRunStatus.RUNNING, updated_at=6.0
        )
        self.store.save_step_run(
            self.principal,
            WorkflowStepRun(
                step_run_id=_id("wstep"),
                run_id=running.run_id,
                node_id="retrieve",
                status=WorkflowStepStatus.RUNNING,
                started_at=6.0,
                finished_at=None,
                input_digest=_digest("question"),
                output_digest=None,
            ),
        )
        self.assertEqual(self.store.recover_interrupted_runs(self.principal, updated_at=7.0), 1)
        self.assertEqual(self.store.get_run(self.principal, run.run_id).status, WorkflowRunStatus.INTERRUPTED)
        self.assertEqual(self.store.list_step_runs(self.principal, run.run_id)[0].status, WorkflowStepStatus.INTERRUPTED)

        waiting = self.store.create_run(
            self.principal,
            WorkflowRun(
                run_id=_id("wrun"), workflow_id=self.workflow.workflow_id, revision_id=revision.revision_id,
                specification_digest=revision.specification_digest, status=WorkflowRunStatus.WAITING_APPROVAL,
                created_at=8.0, updated_at=8.0, created_by=self.principal.subject, input_digest=_digest("second"),
            ),
        )
        approval = self.store.create_approval(
            self.principal,
            WorkflowApproval(
                approval_id=_id("wappr"), run_id=waiting.run_id, node_id="approval",
                requested_at=8.0, requested_by=self.principal.subject,
            ),
        )
        decided = self.store.decide_approval(
            self.principal, approval.approval_id, decision=ApprovalDecision.APPROVED, decided_at=9.0
        )
        self.assertEqual(decided.decision, ApprovalDecision.APPROVED)


if __name__ == "__main__":
    unittest.main()
