"""Strict, storage-neutral Workflow DSL contracts.

This module is the first B0 platform contract.  It deliberately describes only
the bounded native nodes that the platform can audit; a workflow cannot contain
arbitrary code, dynamic imports, shell commands, or provider credentials.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, TypeAlias, TypeVar, cast

from rag_system.application_contracts import validate_model_profile_id
from rag_system.json_contract import JsonContractError, decode_json_object
from rag_system.knowledge_base_contracts import validate_resource_id


WORKFLOW_DSL_SCHEMA_VERSION = 1
MAX_WORKFLOW_INPUTS = 32
MAX_WORKFLOW_NODES = 64
MAX_WORKFLOW_OUTPUTS = 32
MAX_NODE_PARAMETERS = 16

_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}")
_INPUT_REFERENCE_PATTERN = re.compile(r"input\.([a-z][a-z0-9_]{0,63})")
_NODE_REFERENCE_PATTERN = re.compile(
    r"node\.([a-z][a-z0-9_]{0,63})\.([a-z][a-z0-9_]{0,63})"
)

ParameterValue: TypeAlias = str | int | bool


class WorkflowContractError(Exception):
    """Base class for invalid Workflow DSL values."""


class WorkflowValidationError(WorkflowContractError, ValueError):
    """A workflow is unsafe, ambiguous, or outside the supported node set."""


class WorkflowNodeKind(StrEnum):
    """Native operations that will receive audited runtime implementations."""

    KNOWLEDGE_RETRIEVE = "knowledge.retrieve"
    PROMPT_RENDER = "prompt.render"
    MODEL_GENERATE = "model.generate"
    GROUNDING_VALIDATE = "grounding.validate"
    CONDITION = "condition"
    HUMAN_APPROVAL = "human.approval"


class WorkflowResourceKind(StrEnum):
    KNOWLEDGE_BASE = "knowledge_base"
    MODEL_PROFILE = "model_profile"


@dataclass(frozen=True, slots=True)
class WorkflowInput:
    """One named, JSON-compatible workflow input.

    Value schemas are intentionally deferred until the runtime has a typed
    expression and structured-output contract.  Names are still fixed here so
    every reference is statically verifiable.
    """

    name: str
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_name(self.name, "workflow input name"))
        if not isinstance(self.required, bool):
            raise WorkflowValidationError("workflow input required must be a boolean")


@dataclass(frozen=True, slots=True)
class WorkflowResourceRef:
    """A secret-free reference to a platform-managed resource."""

    resource_kind: WorkflowResourceKind
    resource_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.resource_kind, WorkflowResourceKind):
            raise WorkflowValidationError("workflow resource kind is invalid")
        try:
            if self.resource_kind is WorkflowResourceKind.KNOWLEDGE_BASE:
                normalized = validate_resource_id(self.resource_id)
            else:
                normalized = validate_model_profile_id(self.resource_id)
        except ValueError as error:
            raise WorkflowValidationError("workflow resource ID is invalid") from error
        object.__setattr__(self, "resource_id", normalized)


@dataclass(frozen=True, slots=True)
class WorkflowInputBinding:
    """Route one declared input or upstream output into a native node input."""

    target: str
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _validate_name(self.target, "binding target"))
        object.__setattr__(self, "source", _validate_reference(self.source))


@dataclass(frozen=True, slots=True)
class WorkflowNode:
    """One bounded native node and its static data dependencies."""

    node_id: str
    node_kind: WorkflowNodeKind
    depends_on: tuple[str, ...] = ()
    input_bindings: tuple[WorkflowInputBinding, ...] = ()
    output_names: tuple[str, ...] = ()
    resources: tuple[WorkflowResourceRef, ...] = ()
    parameters: Mapping[str, ParameterValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _validate_name(self.node_id, "workflow node ID"))
        if not isinstance(self.node_kind, WorkflowNodeKind):
            raise WorkflowValidationError("workflow node kind is invalid")

        dependencies = _normalize_names(self.depends_on, "workflow dependencies")
        if self.node_id in dependencies:
            raise WorkflowValidationError("a workflow node cannot depend on itself")
        bindings = _normalize_items(
            self.input_bindings, WorkflowInputBinding, "workflow input bindings"
        )
        outputs = _normalize_names(self.output_names, "workflow output names")
        resources = _normalize_items(self.resources, WorkflowResourceRef, "workflow resources")
        parameters = _normalize_parameters(self.parameters)

        if len({binding.target for binding in bindings}) != len(bindings):
            raise WorkflowValidationError("workflow node input bindings cannot share a target")
        if len(set(outputs)) != len(outputs):
            raise WorkflowValidationError("workflow node output names cannot contain duplicates")
        if len({(item.resource_kind, item.resource_id) for item in resources}) != len(resources):
            raise WorkflowValidationError("workflow node resources cannot contain duplicates")

        _validate_node_shape(self.node_kind, bindings, outputs, resources, parameters)
        object.__setattr__(self, "depends_on", dependencies)
        object.__setattr__(
            self, "input_bindings", tuple(sorted(bindings, key=lambda item: item.target))
        )
        object.__setattr__(self, "output_names", tuple(sorted(outputs)))
        object.__setattr__(
            self,
            "resources",
            tuple(sorted(resources, key=lambda item: (item.resource_kind.value, item.resource_id))),
        )
        object.__setattr__(self, "parameters", MappingProxyType(parameters))


@dataclass(frozen=True, slots=True)
class WorkflowOutput:
    """A named public result selected from an upstream node output."""

    name: str
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_name(self.name, "workflow output name"))
        object.__setattr__(self, "source", _validate_reference(self.source))
        if _NODE_REFERENCE_PATTERN.fullmatch(self.source) is None:
            raise WorkflowValidationError("workflow outputs must reference a node output")


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    """An immutable, canonicalizable DAG for an auditable workflow revision."""

    schema_version: int
    inputs: tuple[WorkflowInput, ...]
    nodes: tuple[WorkflowNode, ...]
    outputs: tuple[WorkflowOutput, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != WORKFLOW_DSL_SCHEMA_VERSION
        ):
            raise WorkflowValidationError("workflow DSL schema version is unsupported")
        inputs = _normalize_items(self.inputs, WorkflowInput, "workflow inputs")
        nodes = _normalize_items(self.nodes, WorkflowNode, "workflow nodes")
        outputs = _normalize_items(self.outputs, WorkflowOutput, "workflow outputs")
        if not 1 <= len(inputs) <= MAX_WORKFLOW_INPUTS:
            raise WorkflowValidationError("workflow has an invalid input count")
        if not 1 <= len(nodes) <= MAX_WORKFLOW_NODES:
            raise WorkflowValidationError("workflow has an invalid node count")
        if not 1 <= len(outputs) <= MAX_WORKFLOW_OUTPUTS:
            raise WorkflowValidationError("workflow has an invalid output count")
        if len({item.name for item in inputs}) != len(inputs):
            raise WorkflowValidationError("workflow input names cannot contain duplicates")
        if len({item.node_id for item in nodes}) != len(nodes):
            raise WorkflowValidationError("workflow node IDs cannot contain duplicates")
        if len({item.name for item in outputs}) != len(outputs):
            raise WorkflowValidationError("workflow output names cannot contain duplicates")

        _validate_workflow_graph(inputs, nodes, outputs)
        object.__setattr__(self, "inputs", tuple(sorted(inputs, key=lambda item: item.name)))
        object.__setattr__(self, "nodes", tuple(sorted(nodes, key=lambda item: item.node_id)))
        object.__setattr__(self, "outputs", tuple(sorted(outputs, key=lambda item: item.name)))

    @property
    def digest(self) -> str:
        """Stable SHA-256 over the canonical JSON representation."""

        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "inputs": [
                {"name": item.name, "required": item.required}
                for item in self.inputs
            ],
            "nodes": [
                {
                    "id": item.node_id,
                    "kind": item.node_kind.value,
                    "depends_on": list(item.depends_on),
                    "input_bindings": [
                        {"target": binding.target, "source": binding.source}
                        for binding in item.input_bindings
                    ],
                    "output_names": list(item.output_names),
                    "resources": [
                        {"kind": resource.resource_kind.value, "id": resource.resource_id}
                        for resource in item.resources
                    ],
                    "parameters": dict(item.parameters),
                }
                for item in self.nodes
            ],
            "outputs": [
                {"name": item.name, "source": item.source}
                for item in self.outputs
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, content: str) -> WorkflowSpec:
        try:
            return cls.from_dict(decode_json_object(content))
        except JsonContractError as error:
            raise WorkflowValidationError("workflow DSL must be one strict JSON object") from error

    @classmethod
    def from_dict(cls, value: object) -> WorkflowSpec:
        payload = _require_mapping(value, "workflow DSL")
        _require_exact_keys(
            payload, {"schema_version", "inputs", "nodes", "outputs"}, "workflow DSL"
        )
        return cls(
            schema_version=_require_int(payload["schema_version"], "workflow schema version"),
            inputs=tuple(
                _input_from_dict(item) for item in _require_list(payload["inputs"], "inputs")
            ),
            nodes=tuple(_node_from_dict(item) for item in _require_list(payload["nodes"], "nodes")),
            outputs=tuple(
                _output_from_dict(item)
                for item in _require_list(payload["outputs"], "outputs")
            ),
        )


def _input_from_dict(value: object) -> WorkflowInput:
    payload = _require_mapping(value, "workflow input")
    _require_exact_keys(payload, {"name", "required"}, "workflow input")
    return WorkflowInput(
        name=_require_text(payload["name"], "workflow input name"),
        required=_require_bool(payload["required"], "workflow input required"),
    )


def _node_from_dict(value: object) -> WorkflowNode:
    payload = _require_mapping(value, "workflow node")
    _require_exact_keys(
        payload,
        {"id", "kind", "depends_on", "input_bindings", "output_names", "resources", "parameters"},
        "workflow node",
    )
    bindings = tuple(
        _binding_from_dict(item)
        for item in _require_list(payload["input_bindings"], "input bindings")
    )
    resources = tuple(
        _resource_from_dict(item) for item in _require_list(payload["resources"], "resources")
    )
    parameters = _require_mapping(payload["parameters"], "workflow parameters")
    try:
        kind = WorkflowNodeKind(_require_text(payload["kind"], "workflow node kind"))
    except (TypeError, ValueError) as error:
        raise WorkflowValidationError("workflow node kind is invalid") from error
    return WorkflowNode(
        node_id=_require_text(payload["id"], "workflow node ID"),
        node_kind=kind,
        depends_on=tuple(
            _require_text(item, "workflow dependency")
            for item in _require_list(payload["depends_on"], "dependencies")
        ),
        input_bindings=bindings,
        output_names=tuple(
            _require_text(item, "workflow output name")
            for item in _require_list(payload["output_names"], "output names")
        ),
        resources=resources,
        parameters=_parse_parameters(parameters),
    )


def _binding_from_dict(value: object) -> WorkflowInputBinding:
    payload = _require_mapping(value, "workflow input binding")
    _require_exact_keys(payload, {"target", "source"}, "workflow input binding")
    return WorkflowInputBinding(
        target=_require_text(payload["target"], "binding target"),
        source=_require_text(payload["source"], "binding source"),
    )


def _resource_from_dict(value: object) -> WorkflowResourceRef:
    payload = _require_mapping(value, "workflow resource")
    _require_exact_keys(payload, {"kind", "id"}, "workflow resource")
    try:
        kind = WorkflowResourceKind(_require_text(payload["kind"], "workflow resource kind"))
    except (TypeError, ValueError) as error:
        raise WorkflowValidationError("workflow resource kind is invalid") from error
    return WorkflowResourceRef(
        resource_kind=kind,
        resource_id=_require_text(payload["id"], "workflow resource ID"),
    )


def _output_from_dict(value: object) -> WorkflowOutput:
    payload = _require_mapping(value, "workflow output")
    _require_exact_keys(payload, {"name", "source"}, "workflow output")
    return WorkflowOutput(
        name=_require_text(payload["name"], "workflow output name"),
        source=_require_text(payload["source"], "workflow output source"),
    )


def _validate_workflow_graph(
    inputs: tuple[WorkflowInput, ...],
    nodes: tuple[WorkflowNode, ...],
    outputs: tuple[WorkflowOutput, ...],
) -> None:
    input_names = {item.name for item in inputs}
    node_by_id = {item.node_id: item for item in nodes}
    for node in nodes:
        dependencies = set(node.depends_on)
        if not dependencies <= set(node_by_id):
            raise WorkflowValidationError("workflow node depends on an unknown node")
        referenced_dependencies: set[str] = set()
        for binding in node.input_bindings:
            input_match = _INPUT_REFERENCE_PATTERN.fullmatch(binding.source)
            if input_match is not None:
                if input_match.group(1) not in input_names:
                    raise WorkflowValidationError("workflow binding references an unknown input")
                continue
            node_match = _NODE_REFERENCE_PATTERN.fullmatch(binding.source)
            if node_match is None:
                raise WorkflowValidationError("workflow binding source is invalid")
            source_node_id, source_output = node_match.groups()
            source_node = node_by_id.get(source_node_id)
            if source_node is None or source_output not in source_node.output_names:
                raise WorkflowValidationError("workflow binding references an unknown node output")
            referenced_dependencies.add(source_node_id)
        if dependencies != referenced_dependencies:
            raise WorkflowValidationError(
                "workflow dependencies must exactly match node output bindings"
            )

    _reject_cycles(node_by_id)
    output_nodes: set[str] = set()
    for output in outputs:
        match = _NODE_REFERENCE_PATTERN.fullmatch(output.source)
        if match is None:
            raise WorkflowValidationError("workflow output source is invalid")
        node_id, output_name = match.groups()
        node = node_by_id.get(node_id)
        if node is None or output_name not in node.output_names:
            raise WorkflowValidationError("workflow output references an unknown node output")
        output_nodes.add(node_id)

    reachable: set[str] = set()

    def mark_reachable(node_id: str) -> None:
        if node_id in reachable:
            return
        reachable.add(node_id)
        for dependency in node_by_id[node_id].depends_on:
            mark_reachable(dependency)

    for node_id in output_nodes:
        mark_reachable(node_id)
    if reachable != set(node_by_id):
        raise WorkflowValidationError("workflow cannot contain disconnected nodes")


def _reject_cycles(nodes: Mapping[str, WorkflowNode]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise WorkflowValidationError("workflow graph cannot contain a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in nodes[node_id].depends_on:
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in nodes:
        visit(node_id)


def _validate_node_shape(
    kind: WorkflowNodeKind,
    bindings: tuple[WorkflowInputBinding, ...],
    outputs: tuple[str, ...],
    resources: tuple[WorkflowResourceRef, ...],
    parameters: Mapping[str, ParameterValue],
) -> None:
    expected_inputs = {
        WorkflowNodeKind.KNOWLEDGE_RETRIEVE: {"query"},
        WorkflowNodeKind.PROMPT_RENDER: {"question", "evidence"},
        WorkflowNodeKind.MODEL_GENERATE: {"prompt"},
        WorkflowNodeKind.GROUNDING_VALIDATE: {"answer", "evidence"},
        WorkflowNodeKind.CONDITION: {"validation"},
        WorkflowNodeKind.HUMAN_APPROVAL: {"message"},
    }[kind]
    expected_outputs = {
        WorkflowNodeKind.KNOWLEDGE_RETRIEVE: {"evidence"},
        WorkflowNodeKind.PROMPT_RENDER: {"prompt"},
        WorkflowNodeKind.MODEL_GENERATE: {"answer"},
        WorkflowNodeKind.GROUNDING_VALIDATE: {"validation"},
        WorkflowNodeKind.CONDITION: {"decision"},
        WorkflowNodeKind.HUMAN_APPROVAL: {"decision"},
    }[kind]
    if {item.target for item in bindings} != expected_inputs:
        raise WorkflowValidationError("workflow node has unsupported input bindings")
    if set(outputs) != expected_outputs:
        raise WorkflowValidationError("workflow node has unsupported output names")

    if kind is WorkflowNodeKind.KNOWLEDGE_RETRIEVE:
        _require_single_resource(resources, WorkflowResourceKind.KNOWLEDGE_BASE)
    elif kind is WorkflowNodeKind.MODEL_GENERATE:
        _require_single_resource(resources, WorkflowResourceKind.MODEL_PROFILE)
    elif resources:
        raise WorkflowValidationError("workflow node kind cannot bind platform resources")
    _validate_parameters(kind, parameters)


def _require_single_resource(
    resources: tuple[WorkflowResourceRef, ...], expected_kind: WorkflowResourceKind
) -> None:
    if len(resources) != 1 or resources[0].resource_kind is not expected_kind:
        raise WorkflowValidationError("workflow node has an invalid resource binding")


def _validate_parameters(kind: WorkflowNodeKind, parameters: Mapping[str, ParameterValue]) -> None:
    allowed = {
        WorkflowNodeKind.KNOWLEDGE_RETRIEVE: {"max_results"},
        WorkflowNodeKind.PROMPT_RENDER: {"template"},
        WorkflowNodeKind.MODEL_GENERATE: {"max_output_tokens"},
        WorkflowNodeKind.GROUNDING_VALIDATE: {"require_citations"},
        WorkflowNodeKind.CONDITION: {"rule"},
        WorkflowNodeKind.HUMAN_APPROVAL: {"timeout_seconds"},
    }[kind]
    if not set(parameters) <= allowed:
        raise WorkflowValidationError("workflow node has unsupported parameters")
    if kind is WorkflowNodeKind.PROMPT_RENDER:
        template = parameters.get("template")
        if not isinstance(template, str) or not template.strip():
            raise WorkflowValidationError("prompt.render requires a non-empty template")
    if "max_results" in parameters and not _is_int_between(parameters["max_results"], 1, 20):
        raise WorkflowValidationError("knowledge.retrieve max_results must be between 1 and 20")
    if "max_output_tokens" in parameters and not _is_int_between(
        parameters["max_output_tokens"], 1, 8_192
    ):
        raise WorkflowValidationError("model.generate max_output_tokens must be between 1 and 8192")
    if "require_citations" in parameters and not isinstance(parameters["require_citations"], bool):
        raise WorkflowValidationError("grounding.validate require_citations must be a boolean")
    if "rule" in parameters and parameters["rule"] != "evidence_sufficient":
        raise WorkflowValidationError("condition rule is unsupported")
    if "timeout_seconds" in parameters and not _is_int_between(
        parameters["timeout_seconds"], 60, 604_800
    ):
        raise WorkflowValidationError(
            "human.approval timeout_seconds must be between 60 and 604800"
        )


def _normalize_names(value: object, description: str) -> tuple[str, ...]:
    items = _require_sequence(value, description)
    names = tuple(_validate_name(item, description) for item in items)
    if len(set(names)) != len(names):
        raise WorkflowValidationError(f"{description} cannot contain duplicates")
    return tuple(sorted(names))


T = TypeVar("T")


def _normalize_items(value: object, expected_type: type[T], description: str) -> tuple[T, ...]:
    items = _require_sequence(value, description)
    if any(not isinstance(item, expected_type) for item in items):
        raise WorkflowValidationError(f"{description} contains an invalid item")
    return tuple(cast(T, item) for item in items)


def _normalize_parameters(value: object) -> dict[str, ParameterValue]:
    parameters = _require_mapping(value, "workflow parameters")
    if len(parameters) > MAX_NODE_PARAMETERS:
        raise WorkflowValidationError("workflow node has too many parameters")
    normalized: dict[str, ParameterValue] = {}
    for key, item in parameters.items():
        normalized_key = _validate_name(key, "workflow parameter name")
        if isinstance(item, bool):
            normalized[normalized_key] = item
        elif isinstance(item, int):
            normalized[normalized_key] = item
        elif isinstance(item, str) and len(item) <= 8_000 and _safe_text(item):
            normalized[normalized_key] = item
        else:
            raise WorkflowValidationError("workflow parameter value is invalid")
    return dict(sorted(normalized.items()))


def _parse_parameters(value: Mapping[str, object]) -> dict[str, ParameterValue]:
    """Narrow untrusted JSON values before constructing a typed node."""

    return _normalize_parameters(value)


def _validate_name(value: object, description: str) -> str:
    if not isinstance(value, str) or _NAME_PATTERN.fullmatch(value) is None:
        raise WorkflowValidationError(f"{description} has an invalid format")
    return value


def _validate_reference(value: object) -> str:
    if not isinstance(value, str) or (
        _INPUT_REFERENCE_PATTERN.fullmatch(value) is None
        and _NODE_REFERENCE_PATTERN.fullmatch(value) is None
    ):
        raise WorkflowValidationError("workflow reference has an invalid format")
    return value


def _require_sequence(value: object, description: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise WorkflowValidationError(f"{description} must be a sequence")
    return tuple(value)


def _require_mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise WorkflowValidationError(f"{description} must be an object")
    return value


def _require_text(value: object, description: str) -> str:
    if not isinstance(value, str):
        raise WorkflowValidationError(f"{description} must be text")
    return value


def _require_bool(value: object, description: str) -> bool:
    if not isinstance(value, bool):
        raise WorkflowValidationError(f"{description} must be a boolean")
    return value


def _require_int(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkflowValidationError(f"{description} must be an integer")
    return value


def _require_list(value: object, description: str) -> list[object]:
    if not isinstance(value, list):
        raise WorkflowValidationError(f"{description} must be an array")
    return value


def _require_exact_keys(
    payload: Mapping[str, object], expected: set[str], description: str
) -> None:
    if set(payload) != expected:
        raise WorkflowValidationError(f"{description} has an invalid shape")


def _is_int_between(value: ParameterValue, minimum: int, maximum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum


def _safe_text(value: str) -> bool:
    return all(ord(character) >= 32 or character in {"\n", "\r", "\t"} for character in value)


__all__ = [
    "MAX_NODE_PARAMETERS",
    "MAX_WORKFLOW_INPUTS",
    "MAX_WORKFLOW_NODES",
    "MAX_WORKFLOW_OUTPUTS",
    "WORKFLOW_DSL_SCHEMA_VERSION",
    "WorkflowContractError",
    "WorkflowInput",
    "WorkflowInputBinding",
    "WorkflowNode",
    "WorkflowNodeKind",
    "WorkflowOutput",
    "WorkflowResourceKind",
    "WorkflowResourceRef",
    "WorkflowSpec",
    "WorkflowValidationError",
]
