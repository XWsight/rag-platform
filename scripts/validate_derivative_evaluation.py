"""Validate governance metadata for a derivative project's domain evaluations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from rag_system.answer_suite import load_answer_suite, validate_answer_suite_contract
from rag_system.benchmark_suite import load_retrieval_suite, validate_suite_contract
from rag_system.evaluation import DatasetValidationError
from rag_system.evaluation_suite import EvaluationSuiteError


_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "product_name",
        "base_revision",
        "owner",
        "data_classification",
        "status",
        "held_out_test_status",
        "retrieval",
        "answer",
    }
)
_SUITE_FIELDS = frozenset({"suite", "contract"})
_REVISION_PATTERN = re.compile(r"(?:unrecorded|[0-9a-f]{7,64})")
_TEXT_LIMIT = 160


class DerivativeEvaluationError(ValueError):
    """The derivative evaluation-governance document is incomplete or unsafe."""


def load_governance(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_text(encoding="utf-8")
    except OSError as error:
        raise DerivativeEvaluationError("governance file cannot be read") from error
    try:
        payload = json.loads(raw, object_pairs_hook=_strict_object)
    except (json.JSONDecodeError, ValueError) as error:
        raise DerivativeEvaluationError("governance file must be strict JSON") from error
    if not isinstance(payload, dict) or frozenset(payload) != _ROOT_FIELDS:
        raise DerivativeEvaluationError("governance fields are invalid")
    if payload["schema_version"] != 1:
        raise DerivativeEvaluationError("governance schema_version is unsupported")
    for field in ("product_name", "owner"):
        _text(payload[field], field)
    revision = payload["base_revision"]
    if not isinstance(revision, str) or _REVISION_PATTERN.fullmatch(revision) is None:
        raise DerivativeEvaluationError("base_revision is invalid")
    if payload["data_classification"] not in {"public", "internal", "confidential"}:
        raise DerivativeEvaluationError("data_classification is invalid")
    if payload["status"] not in {"draft", "ready"}:
        raise DerivativeEvaluationError("status is invalid")
    if payload["held_out_test_status"] not in {"unconsumed", "consumed"}:
        raise DerivativeEvaluationError("held_out_test_status is invalid")
    for name in ("retrieval", "answer"):
        descriptor = payload[name]
        if not isinstance(descriptor, dict) or frozenset(descriptor) != _SUITE_FIELDS:
            raise DerivativeEvaluationError(f"{name} suite fields are invalid")
        for field in _SUITE_FIELDS:
            _relative_path(descriptor[field], field)
    return payload


def validate_governance(path: Path, *, require_ready: bool = False) -> dict[str, object]:
    """Validate metadata, and validate governed suites when release-ready."""

    governance = load_governance(path)
    status = governance["status"]
    if status == "draft":
        if require_ready:
            raise DerivativeEvaluationError("domain evaluation governance is still draft")
        return {"status": "draft", "validated_suites": False}
    if str(governance["owner"]).startswith("replace-with-"):
        raise DerivativeEvaluationError("domain evaluation owner is still a template placeholder")
    if governance["held_out_test_status"] != "unconsumed":
        raise DerivativeEvaluationError("held-out test data has already been consumed")

    root = path.expanduser().resolve().parent
    retrieval = governance["retrieval"]
    answer = governance["answer"]
    assert isinstance(retrieval, dict) and isinstance(answer, dict)
    try:
        retrieval_suite = load_retrieval_suite(_inside(root, retrieval["suite"]))
        validate_suite_contract(retrieval_suite, _inside(root, retrieval["contract"]))
        answer_suite = load_answer_suite(_inside(root, answer["suite"]))
        validate_answer_suite_contract(answer_suite, _inside(root, answer["contract"]))
    except (DatasetValidationError, EvaluationSuiteError, OSError) as error:
        raise DerivativeEvaluationError("domain evaluation suites are invalid") from error
    if not retrieval_suite.cases_for_split("test") or not answer_suite.cases_for_split("test"):
        raise DerivativeEvaluationError("each governed suite requires a held-out test split")
    return {
        "status": "ready",
        "validated_suites": True,
        "retrieval_cases": len(retrieval_suite.cases),
        "answer_cases": len(answer_suite.cases),
    }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= _TEXT_LIMIT:
        raise DerivativeEvaluationError(f"{field} is invalid")
    return value


def _relative_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value or len(value) > 240:
        raise DerivativeEvaluationError(f"{field} path is invalid")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise DerivativeEvaluationError(f"{field} path must remain inside the evaluation directory")
    return candidate


def _inside(root: Path, value: object) -> Path:
    candidate = (root / _relative_path(value, "suite")).resolve()
    if root != candidate and root not in candidate.parents:
        raise DerivativeEvaluationError("evaluation path escapes its directory")
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("governance", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        result = validate_governance(arguments.governance, require_ready=arguments.require_ready)
    except DerivativeEvaluationError as error:
        print(f"derivative evaluation rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
