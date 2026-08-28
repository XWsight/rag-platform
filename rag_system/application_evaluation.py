"""Auditable evaluation evidence bound to one immutable application revision."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from rag_system.answer_benchmark import AnswerBenchmarkReport
from rag_system.application_contracts import (
    ApplicationRevision,
    ApplicationValidationError,
    is_valid_timestamp,
    validate_application_id,
    validate_revision_id,
)


class ApplicationEvaluationError(ValueError):
    """Application evaluation evidence is incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class ApplicationEvaluationReport:
    application_id: str
    revision_id: str
    revision_number: int
    configuration_digest: str
    generated_at: float
    benchmark: AnswerBenchmarkReport

    def __post_init__(self) -> None:
        try:
            validate_application_id(self.application_id)
            validate_revision_id(self.revision_id)
        except ApplicationValidationError as error:
            raise ApplicationEvaluationError(str(error)) from error
        if (
            isinstance(self.revision_number, bool)
            or not isinstance(self.revision_number, int)
            or self.revision_number < 1
        ):
            raise ApplicationEvaluationError("revision_number must be a positive integer")
        if not is_valid_timestamp(self.generated_at):
            raise ApplicationEvaluationError("generated_at must be finite and non-negative")
        if not isinstance(self.benchmark, AnswerBenchmarkReport):
            raise ApplicationEvaluationError("benchmark must use the typed report contract")
        if len(self.configuration_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.configuration_digest
        ):
            raise ApplicationEvaluationError("configuration_digest must be a SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "application_id": self.application_id,
            "revision_id": self.revision_id,
            "revision_number": self.revision_number,
            "configuration_digest": self.configuration_digest,
            "generated_at": self.generated_at,
            "benchmark": self.benchmark.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def bind_application_evaluation(
    revision: ApplicationRevision,
    benchmark: AnswerBenchmarkReport,
    *,
    generated_at: float,
) -> ApplicationEvaluationReport:
    if not isinstance(revision, ApplicationRevision) or not isinstance(
        benchmark, AnswerBenchmarkReport
    ):
        raise ApplicationEvaluationError("revision and benchmark must use typed contracts")
    policy = revision.configuration.answer_policy
    session = revision.configuration.session_policy
    configuration = {
        "schema_version": revision.configuration_schema_version,
        "knowledge_base_ids": list(revision.configuration.knowledge_base_ids),
        "retrieval_profile": revision.configuration.retrieval_profile.value,
        "answer_policy": {
            "require_citations": policy.require_citations,
            "allow_cloud": policy.allow_cloud,
            "allow_web": policy.allow_web,
            "allow_research": policy.allow_research,
        },
        "session_policy": {"enabled": session.enabled, "ttl_seconds": session.ttl_seconds},
    }
    encoded = json.dumps(configuration, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return ApplicationEvaluationReport(
        application_id=revision.application_id,
        revision_id=revision.revision_id,
        revision_number=revision.revision_number,
        configuration_digest=hashlib.sha256(encoded.encode()).hexdigest(),
        generated_at=generated_at,
        benchmark=benchmark,
    )


__all__ = [
    "ApplicationEvaluationError",
    "ApplicationEvaluationReport",
    "bind_application_evaluation",
]
