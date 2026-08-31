"""Safe built-in implementations for deterministic workflow node kinds.

Resource-backed retrieval and generation remain explicit composition-root
ports.  These helpers only implement operations that can be performed without
provider selection, credentials, or arbitrary expression evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rag_system.tenancy import Principal
from rag_system.workflow_contracts import WorkflowNode, WorkflowNodeKind
from rag_system.workflow_runtime import NativeWorkflowNodeExecutor, WorkflowNodeExecutionError


def built_in_node_executors(
    *,
    retrieve: NativeWorkflowNodeExecutor,
    generate: NativeWorkflowNodeExecutor,
) -> dict[WorkflowNodeKind, NativeWorkflowNodeExecutor]:
    """Return the closed native node set for one configured runtime profile."""

    if not callable(retrieve) or not callable(generate):
        raise TypeError("retrieve and generate executors must be callable")
    return {
        WorkflowNodeKind.KNOWLEDGE_RETRIEVE: retrieve,
        WorkflowNodeKind.PROMPT_RENDER: render_prompt,
        WorkflowNodeKind.MODEL_GENERATE: generate,
        WorkflowNodeKind.GROUNDING_VALIDATE: validate_grounding,
        WorkflowNodeKind.CONDITION: evaluate_condition,
    }


def render_prompt(
    _principal: Principal, node: WorkflowNode, values: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Perform literal placeholder substitution, never template evaluation."""

    _require_kind(node, WorkflowNodeKind.PROMPT_RENDER)
    template = node.parameters.get("template")
    if not isinstance(template, str):
        raise WorkflowNodeExecutionError("prompt_template_invalid")
    question = _text(values.get("question"))
    evidence = _render_evidence(values.get("evidence"))
    return {"prompt": template.replace("{{ question }}", question).replace("{{ evidence }}", evidence)}


def validate_grounding(
    _principal: Principal, node: WorkflowNode, values: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Produce a simple, explainable gate used by the fixed condition node."""

    _require_kind(node, WorkflowNodeKind.GROUNDING_VALIDATE)
    answer = _text(values.get("answer")).strip()
    evidence = values.get("evidence")
    has_evidence = _evidence_present(evidence)
    require_citations = bool(node.parameters.get("require_citations", False))
    return {
        "validation": {
            "answer_present": bool(answer),
            "evidence_present": has_evidence,
            "evidence_sufficient": bool(answer) and (has_evidence or not require_citations),
        }
    }


def evaluate_condition(
    _principal: Principal, node: WorkflowNode, values: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Evaluate only the statically declared ``evidence_sufficient`` rule."""

    _require_kind(node, WorkflowNodeKind.CONDITION)
    validation = values.get("validation")
    if not isinstance(validation, Mapping) or not isinstance(validation.get("evidence_sufficient"), bool):
        raise WorkflowNodeExecutionError("condition_input_invalid")
    return {"decision": "allow" if validation["evidence_sufficient"] else "refuse"}


def _require_kind(node: WorkflowNode, expected: WorkflowNodeKind) -> None:
    if not isinstance(node, WorkflowNode) or node.node_kind is not expected:
        raise WorkflowNodeExecutionError("native_node_contract_invalid")


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise WorkflowNodeExecutionError("native_node_input_invalid")
    return value


def _render_evidence(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_text(item) for item in value)
    if isinstance(value, Mapping):
        return "\n".join(f"{key}: {item}" for key, item in sorted(value.items()) if isinstance(key, str))
    raise WorkflowNodeExecutionError("native_node_input_invalid")


def _evidence_present(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, Mapping)):
        return bool(value)
    return False


__all__ = [
    "built_in_node_executors",
    "evaluate_condition",
    "render_prompt",
    "validate_grounding",
]
