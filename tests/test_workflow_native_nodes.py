from __future__ import annotations

import unittest

from rag_system.tenancy import Principal, TenantId
from rag_system.workflow_contracts import WorkflowInputBinding, WorkflowNode, WorkflowNodeKind
from rag_system.workflow_native_nodes import evaluate_condition, render_prompt, validate_grounding


PRINCIPAL = Principal("writer", TenantId("tenant-a"), frozenset({"writer"}))


class NativeWorkflowNodeTests(unittest.TestCase):
    def test_prompt_grounding_and_condition_are_deterministic(self) -> None:
        prompt = WorkflowNode(
            "prompt", WorkflowNodeKind.PROMPT_RENDER,
            input_bindings=(
                WorkflowInputBinding("question", "input.question"),
                WorkflowInputBinding("evidence", "input.evidence"),
            ), output_names=("prompt",), parameters={"template": "Q={{ question }} E={{ evidence }}"},
        )
        validation = WorkflowNode(
            "validate", WorkflowNodeKind.GROUNDING_VALIDATE,
            input_bindings=(
                WorkflowInputBinding("answer", "input.answer"),
                WorkflowInputBinding("evidence", "input.evidence"),
            ), output_names=("validation",), parameters={"require_citations": True},
        )
        condition = WorkflowNode(
            "condition", WorkflowNodeKind.CONDITION,
            input_bindings=(WorkflowInputBinding("validation", "input.validation"),),
            output_names=("decision",), parameters={"rule": "evidence_sufficient"},
        )

        rendered = render_prompt(PRINCIPAL, prompt, {"question": "q", "evidence": ["e1", "e2"]})
        checked = validate_grounding(PRINCIPAL, validation, {"answer": "a", "evidence": ["e1"]})

        self.assertEqual(rendered, {"prompt": "Q=q E=e1\ne2"})
        self.assertEqual(evaluate_condition(PRINCIPAL, condition, checked), {"decision": "allow"})


if __name__ == "__main__":
    unittest.main()
