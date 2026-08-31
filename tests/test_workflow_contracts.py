from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError

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
    WorkflowValidationError,
)


KNOWLEDGE_BASE_ID = "kb_12345678901234567890123456789012"


def _binding(target: str, source: str) -> WorkflowInputBinding:
    return WorkflowInputBinding(target=target, source=source)


def _workflow(*, reverse_order: bool = False) -> WorkflowSpec:
    retrieve = WorkflowNode(
        node_id="retrieve",
        node_kind=WorkflowNodeKind.KNOWLEDGE_RETRIEVE,
        input_bindings=(_binding("query", "input.question"),),
        output_names=("evidence",),
        resources=(
            WorkflowResourceRef(WorkflowResourceKind.KNOWLEDGE_BASE, KNOWLEDGE_BASE_ID),
        ),
        parameters={"max_results": 5},
    )
    prompt = WorkflowNode(
        node_id="prompt",
        node_kind=WorkflowNodeKind.PROMPT_RENDER,
        depends_on=("retrieve",),
        input_bindings=(
            _binding("evidence", "node.retrieve.evidence"),
            _binding("question", "input.question"),
        ),
        output_names=("prompt",),
        parameters={"template": "Question: {{ question }}\nEvidence: {{ evidence }}"},
    )
    generate = WorkflowNode(
        node_id="generate",
        node_kind=WorkflowNodeKind.MODEL_GENERATE,
        depends_on=("prompt",),
        input_bindings=(_binding("prompt", "node.prompt.prompt"),),
        output_names=("answer",),
        resources=(WorkflowResourceRef(WorkflowResourceKind.MODEL_PROFILE, "default"),),
        parameters={"max_output_tokens": 1_024},
    )
    validate = WorkflowNode(
        node_id="validate",
        node_kind=WorkflowNodeKind.GROUNDING_VALIDATE,
        depends_on=("generate", "retrieve"),
        input_bindings=(
            _binding("answer", "node.generate.answer"),
            _binding("evidence", "node.retrieve.evidence"),
        ),
        output_names=("validation",),
        parameters={"require_citations": True},
    )
    condition = WorkflowNode(
        node_id="condition",
        node_kind=WorkflowNodeKind.CONDITION,
        depends_on=("validate",),
        input_bindings=(_binding("validation", "node.validate.validation"),),
        output_names=("decision",),
        parameters={"rule": "evidence_sufficient"},
    )
    nodes = (retrieve, prompt, generate, validate, condition)
    if reverse_order:
        nodes = tuple(reversed(nodes))
    return WorkflowSpec(
        schema_version=WORKFLOW_DSL_SCHEMA_VERSION,
        inputs=(WorkflowInput("question"),),
        nodes=nodes,
        outputs=(
            WorkflowOutput("answer", "node.generate.answer"),
            WorkflowOutput("decision", "node.condition.decision"),
        ),
    )


class WorkflowContractTests(unittest.TestCase):
    def test_valid_workflow_is_immutable_and_has_a_canonical_digest(self) -> None:
        workflow = _workflow()

        self.assertEqual(workflow.nodes[0].node_id, "condition")
        self.assertEqual(workflow.digest, _workflow(reverse_order=True).digest)
        self.assertEqual(WorkflowSpec.from_json(workflow.to_json()), workflow)
        with self.assertRaises(FrozenInstanceError):
            workflow.nodes = ()  # type: ignore[misc]
        with self.assertRaises(TypeError):
            workflow.nodes[0].parameters["max_results"] = 1  # type: ignore[index]

    def test_json_decoder_rejects_unknown_and_duplicate_fields(self) -> None:
        payload = _workflow().to_dict()
        payload["unexpected"] = True
        with self.assertRaises(WorkflowValidationError):
            WorkflowSpec.from_json(json.dumps(payload))
        duplicate = _workflow().to_json().replace(
            '"schema_version":1', '"schema_version":1,"schema_version":1', 1
        )
        with self.assertRaises(WorkflowValidationError):
            WorkflowSpec.from_json(duplicate)

    def test_node_shape_rejects_unbounded_or_incorrect_capabilities(self) -> None:
        with self.assertRaises(WorkflowValidationError):
            WorkflowNode(
                node_id="generate",
                node_kind=WorkflowNodeKind.MODEL_GENERATE,
                input_bindings=(_binding("prompt", "input.question"),),
                output_names=("answer",),
                resources=(),
            )
        with self.assertRaises(WorkflowValidationError):
            WorkflowNode(
                node_id="prompt",
                node_kind=WorkflowNodeKind.PROMPT_RENDER,
                input_bindings=(
                    _binding("question", "input.question"),
                    _binding("evidence", "input.question"),
                ),
                output_names=("prompt",),
                parameters={"template": "ok", "shell": "powershell"},
            )
        with self.assertRaises(WorkflowValidationError):
            WorkflowNode(
                node_id="condition",
                node_kind=WorkflowNodeKind.CONDITION,
                input_bindings=(_binding("validation", "input.question"),),
                output_names=("decision",),
                parameters={"rule": "eval(user_code)"},
            )

    def test_graph_requires_declared_inputs_direct_dependencies_and_reachable_nodes(self) -> None:
        retrieve = WorkflowNode(
            node_id="retrieve",
            node_kind=WorkflowNodeKind.KNOWLEDGE_RETRIEVE,
            input_bindings=(_binding("query", "input.question"),),
            output_names=("evidence",),
            resources=(
                WorkflowResourceRef(WorkflowResourceKind.KNOWLEDGE_BASE, KNOWLEDGE_BASE_ID),
            ),
        )
        prompt = WorkflowNode(
            node_id="prompt",
            node_kind=WorkflowNodeKind.PROMPT_RENDER,
            depends_on=(),
            input_bindings=(
                _binding("question", "input.question"),
                _binding("evidence", "node.retrieve.evidence"),
            ),
            output_names=("prompt",),
            parameters={"template": "{{ question }}"},
        )
        with self.assertRaisesRegex(WorkflowValidationError, "dependencies"):
            WorkflowSpec(
                schema_version=WORKFLOW_DSL_SCHEMA_VERSION,
                inputs=(WorkflowInput("question"),),
                nodes=(retrieve, prompt),
                outputs=(WorkflowOutput("prompt", "node.prompt.prompt"),),
            )

        orphan = WorkflowNode(
            node_id="orphan",
            node_kind=WorkflowNodeKind.KNOWLEDGE_RETRIEVE,
            input_bindings=(_binding("query", "input.question"),),
            output_names=("evidence",),
            resources=(
                WorkflowResourceRef(WorkflowResourceKind.KNOWLEDGE_BASE, KNOWLEDGE_BASE_ID),
            ),
        )
        workflow = _workflow()
        with self.assertRaisesRegex(WorkflowValidationError, "disconnected"):
            WorkflowSpec(
                schema_version=workflow.schema_version,
                inputs=workflow.inputs,
                nodes=(*workflow.nodes, orphan),
                outputs=workflow.outputs,
            )

    def test_workflow_rejects_invalid_resources_parameters_and_output_references(self) -> None:
        with self.assertRaises(WorkflowValidationError):
            WorkflowResourceRef(WorkflowResourceKind.MODEL_PROFILE, "https://model.example")
        with self.assertRaises(WorkflowValidationError):
            WorkflowNode(
                node_id="retrieve",
                node_kind=WorkflowNodeKind.KNOWLEDGE_RETRIEVE,
                input_bindings=(_binding("query", "input.question"),),
                output_names=("evidence",),
                resources=(
                    WorkflowResourceRef(WorkflowResourceKind.KNOWLEDGE_BASE, KNOWLEDGE_BASE_ID),
                ),
                parameters={"max_results": True},
            )
        with self.assertRaises(WorkflowValidationError):
            WorkflowOutput("answer", "input.question")


if __name__ == "__main__":
    unittest.main()
