"""Auditable evaluation evidence bound to one immutable application revision."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from rag_system.answer_benchmark import (
    AnswerBenchmarkMetrics,
    AnswerBenchmarkReport,
    AnswerCaseResult,
)
from rag_system.application_contracts import (
    ApplicationRevision,
    ApplicationValidationError,
    is_valid_timestamp,
    validate_application_id,
    validate_revision_id,
)
from rag_system.domain import AnswerClaim


MAX_EVALUATION_REPORT_BYTES = 4 * 1024 * 1024


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

    @classmethod
    def from_json(cls, value: object) -> ApplicationEvaluationReport:
        """Restore a previously persisted report while rejecting altered evidence."""

        if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_EVALUATION_REPORT_BYTES:
            raise ApplicationEvaluationError("evaluation report must be JSON text")
        try:
            payload = json.loads(value)
            if not isinstance(payload, dict) or set(payload) != {
                "schema_version", "application_id", "revision_id", "revision_number",
                "configuration_digest", "generated_at", "benchmark",
            } or payload["schema_version"] != 1:
                raise ValueError
            benchmark_payload = payload["benchmark"]
            if not isinstance(benchmark_payload, dict) or set(benchmark_payload) != {
                "schema_version", "dataset_digest", "case_count", "fact_count", "metrics", "results"
            } or benchmark_payload["schema_version"] != 1:
                raise ValueError
            metrics = benchmark_payload["metrics"]
            if not isinstance(metrics, dict) or set(metrics) != {
                "contract_success_rate", "refusal_accuracy", "fact_recall", "atomic_claim_rate",
                "attribution_precision",
            }:
                raise ValueError
            results_payload = benchmark_payload["results"]
            if not isinstance(results_payload, list):
                raise ValueError
            results = tuple(_result_from_dict(item) for item in results_payload)
            benchmark = AnswerBenchmarkReport(
                dataset_digest=benchmark_payload["dataset_digest"],
                case_count=benchmark_payload["case_count"],
                fact_count=benchmark_payload["fact_count"],
                metrics=AnswerBenchmarkMetrics(**metrics),
                results=results,
            )
            return cls(
                application_id=payload["application_id"],
                revision_id=payload["revision_id"],
                revision_number=payload["revision_number"],
                configuration_digest=payload["configuration_digest"],
                generated_at=payload["generated_at"],
                benchmark=benchmark,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ApplicationEvaluationError("evaluation report has an invalid contract") from error


def _result_from_dict(value: object) -> AnswerCaseResult:
    if not isinstance(value, dict) or set(value) != {
        "case_id", "contract_valid", "refusal_correct", "expected_fact_count",
        "recovered_fact_ids", "claim_count", "atomic_claim_count", "grounded_claim_count",
        "claims", "error_code", "passed",
    }:
        raise ValueError
    claims = value["claims"]
    if not isinstance(claims, list):
        raise ValueError
    resolved_claims = tuple(
        AnswerClaim(text=item["text"], citation_ids=tuple(item["citation_ids"]))
        for item in claims
        if isinstance(item, dict) and set(item) == {"text", "citation_ids"}
        and isinstance(item["citation_ids"], list)
    )
    if len(resolved_claims) != len(claims):
        raise ValueError
    result = AnswerCaseResult(
        case_id=value["case_id"], contract_valid=value["contract_valid"],
        refusal_correct=value["refusal_correct"], expected_fact_count=value["expected_fact_count"],
        recovered_fact_ids=tuple(value["recovered_fact_ids"]), claim_count=value["claim_count"],
        atomic_claim_count=value["atomic_claim_count"], grounded_claim_count=value["grounded_claim_count"],
        claims=resolved_claims, error_code=value["error_code"],
    )
    if value["passed"] is not result.passed:
        raise ValueError
    return result


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
        "model_profile_id": revision.configuration.model_profile_id,
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
    "MAX_EVALUATION_REPORT_BYTES",
    "bind_application_evaluation",
]
