"""Turn a measured local-index microbenchmark into a conservative scale decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def load_benchmark(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("benchmark report cannot be read as JSON") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 3:
        raise ValueError("benchmark report schema is unsupported")
    if not isinstance(payload.get("scope"), str) or not isinstance(payload.get("environment"), dict):
        raise ValueError("benchmark report metadata is invalid")
    if not isinstance(payload.get("results"), list):
        raise ValueError("benchmark report results are invalid")
    return payload


def assess_capacity(
    benchmark: dict[str, Any], *, target_chunks: int, p95_budget_ms: float
) -> dict[str, object]:
    """Return a non-deploying decision based only on one measured target size."""

    if target_chunks < 1 or p95_budget_ms <= 0:
        raise ValueError("capacity targets must be positive")
    rows = benchmark.get("results")
    if not isinstance(rows, list):
        raise ValueError("benchmark report results are invalid")

    measured: dict[int, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("benchmark report row is invalid")
        chunk_count = row.get("chunk_count")
        p95_ms = row.get("p95_ms")
        if (
            not isinstance(chunk_count, int)
            or isinstance(chunk_count, bool)
            or chunk_count < 1
            or not isinstance(p95_ms, (int, float))
            or isinstance(p95_ms, bool)
            or p95_ms < 0
            or chunk_count in measured
        ):
            raise ValueError("benchmark report row is invalid")
        measured[chunk_count] = float(p95_ms)

    source = benchmark["environment"]
    assert isinstance(source, dict)
    result: dict[str, object] = {
        "schema_version": 1,
        "scope": "microbenchmark triage only; not an end-to-end or deployment decision",
        "source_revision": source.get("source_revision"),
        "working_tree_clean": source.get("working_tree_clean"),
        "target_chunks": target_chunks,
        "p95_budget_ms": p95_budget_ms,
        "measured_chunk_counts": sorted(measured),
    }
    measured_p95 = measured.get(target_chunks)
    if measured_p95 is None:
        result.update(
            {
                "decision": "measure_target",
                "reason": "The exact target chunk count has not been benchmarked.",
            }
        )
        return result

    result["measured_p95_ms"] = measured_p95
    if measured_p95 <= p95_budget_ms:
        result.update(
            {
                "decision": "keep_local_exact_candidate",
                "reason": "Measured exact-search P95 is within the supplied microbenchmark budget.",
            }
        )
    else:
        result.update(
            {
                "decision": "evaluate_ann_candidate",
                "reason": "Measured exact-search P95 exceeds the supplied microbenchmark budget.",
            }
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path, help="schema-v3 benchmark JSON from benchmark_local_index.py")
    parser.add_argument("--target-chunks", required=True, type=_positive_int)
    parser.add_argument("--p95-budget-ms", required=True, type=_positive_float)
    parser.add_argument("--json-output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        result = assess_capacity(
            load_benchmark(arguments.benchmark),
            target_chunks=arguments.target_chunks,
            p95_budget_ms=arguments.p95_budget_ms,
        )
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if arguments.json_output:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
