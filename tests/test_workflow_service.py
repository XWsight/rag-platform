from __future__ import annotations

import secrets
import tempfile
import unittest
from pathlib import Path

from rag_system.knowledge_base_contracts import KnowledgeBaseStatus
from rag_system.tenancy import Principal, TenantId
from rag_system.workflow_contracts import (
    WORKFLOW_DSL_SCHEMA_VERSION, WorkflowInput, WorkflowInputBinding, WorkflowNode,
    WorkflowNodeKind, WorkflowOutput, WorkflowResourceKind, WorkflowResourceRef, WorkflowSpec,
)
from rag_system.workflow_models import ExecutionBudget, WorkflowEvaluation
from rag_system.workflow_service import WorkflowService, WorkflowServiceValidationError
from rag_system.workflow_store import WorkflowStore


PROJECT_ID = "prj_12345678901234567890123456789012"
KNOWLEDGE_BASE_ID = "kb_12345678901234567890123456789012"


class _Projects:
    def get_project(self, _principal, project_id):
        if project_id != PROJECT_ID:
            raise ValueError
        return object()


class _KnowledgeBases:
    def get(self, _principal, resource_id):
        if resource_id != KNOWLEDGE_BASE_ID:
            raise ValueError
        return type("Record", (), {"status": KnowledgeBaseStatus.READY})()


def _specification() -> WorkflowSpec:
    retrieve = WorkflowNode(
        node_id="retrieve", node_kind=WorkflowNodeKind.KNOWLEDGE_RETRIEVE,
        input_bindings=(WorkflowInputBinding("query", "input.question"),), output_names=("evidence",),
        resources=(WorkflowResourceRef(WorkflowResourceKind.KNOWLEDGE_BASE, KNOWLEDGE_BASE_ID),),
    )
    prompt = WorkflowNode(
        node_id="prompt", node_kind=WorkflowNodeKind.PROMPT_RENDER, depends_on=("retrieve",),
        input_bindings=(WorkflowInputBinding("question", "input.question"), WorkflowInputBinding("evidence", "node.retrieve.evidence")),
        output_names=("prompt",), parameters={"template": "{{ question }}"},
    )
    generate = WorkflowNode(
        node_id="generate", node_kind=WorkflowNodeKind.MODEL_GENERATE, depends_on=("prompt",),
        input_bindings=(WorkflowInputBinding("prompt", "node.prompt.prompt"),), output_names=("answer",),
        resources=(WorkflowResourceRef(WorkflowResourceKind.MODEL_PROFILE, "default"),),
    )
    return WorkflowSpec(WORKFLOW_DSL_SCHEMA_VERSION, (WorkflowInput("question"),), (retrieve, prompt, generate), (WorkflowOutput("answer", "node.generate.answer"),))


class WorkflowServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = WorkflowStore(Path(self.tempdir.name) / "workflows.sqlite3")
        self.principal = Principal("operator", TenantId("tenant-a"), frozenset({"reader", "writer", "operator"}))
        self.service = WorkflowService(self.store, _Projects(), _KnowledgeBases(), clock=lambda: 1.0)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_publishing_requires_passing_evaluation_bound_to_revision(self) -> None:
        workflow = self.service.create_workflow(self.principal, PROJECT_ID, "Workflow")
        draft = self.service.update_draft(
            self.principal, workflow.workflow_id, _specification(), ExecutionBudget(),
            expected_version=0, change_summary="Initial",
        )
        revision = self.service.create_revision_from_draft(
            self.principal, workflow.workflow_id, expected_version=draft.version
        )
        with self.assertRaisesRegex(WorkflowServiceValidationError, "evaluation"):
            self.service.publish(self.principal, workflow.workflow_id, revision.revision_id, expected_active_revision_id=None)

        evaluation = WorkflowEvaluation(
            evaluation_id=f"weval_{secrets.token_hex(16)}", workflow_id=workflow.workflow_id,
            revision_id=revision.revision_id, specification_digest=revision.specification_digest,
            generated_at=1.0, case_count=2, passed_case_count=2,
        )
        self.service.record_evaluation(self.principal, evaluation)
        published = self.service.publish(
            self.principal, workflow.workflow_id, revision.revision_id, expected_active_revision_id=None
        )
        self.assertEqual(published.workflow.active_revision_id, revision.revision_id)


if __name__ == "__main__":
    unittest.main()
