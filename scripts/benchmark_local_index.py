"""Measure exact local-vector search without downloading an embedding model.

This microbenchmark exercises ``LocalVectorIndex.search`` only.  It excludes
parsing, embedding, persistence, BM25, reranking, network I/O, and concurrency.
It exposes the linear cost of the in-process exact-cosine backend; it is not a
production latency claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.domain import Chunk, IndexRef  # noqa: E402
from rag_system.retrieval import LocalVectorIndex  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(_positive_int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers") from error
    if not sizes:
        raise argparse.ArgumentTypeError("at least one size is required")
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("sizes must not contain duplicates")
    return sizes


def percentile(samples_ns: Iterable[int], fraction: float) -> int:
    ordered = sorted(samples_ns)
    if not ordered:
        raise ValueError("cannot calculate a percentile of no samples")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def build_synthetic_index(chunk_count: int, dimension: int) -> LocalVectorIndex:
    """Build deterministic non-zero vectors without involving an Embedder."""

    chunks: list[Chunk] = []
    vectors: dict[str, tuple[float, ...]] = {}
    for number in range(chunk_count):
        chunk_id = f"chunk{number:08d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=f"document{number // 8:06d}",
                source_name=f"synthetic-{number // 8:06d}.txt",
                text="synthetic benchmark chunk",
                chunk_index=number % 8,
                start_char=0,
                end_char=25,
            )
        )
        vectors[chunk_id] = tuple(
            float(((number + 1) * (axis + 3)) % 97 + 1) for axis in range(dimension)
        )
    query = tuple(float((axis * 7) % 89 + 1) for axis in range(dimension))
    return LocalVectorIndex(
        vectors=vectors,
        index_ref=IndexRef(
            index_id="idx_benchmark",
            document_count=(chunk_count + 7) // 8,
            chunk_count=chunk_count,
            created_at=0.0,
        ),
        chunks=chunks,
        persistent=False,
        embed_query=lambda _: query,
        delete_persisted=lambda: False,
    )


def benchmark_index(
    index: LocalVectorIndex, *, queries: int, warmup: int, top_k: int
) -> dict[str, float | int]:
    for _ in range(warmup):
        index.search("synthetic query", top_k=top_k)

    samples_ns: list[int] = []
    for _ in range(queries):
        started_ns = time.perf_counter_ns()
        hits = index.search("synthetic query", top_k=top_k)
        elapsed_ns = time.perf_counter_ns() - started_ns
        if len(hits) != top_k:
            raise RuntimeError("synthetic index did not return the requested top-k")
        samples_ns.append(elapsed_ns)

    return {
        "queries": queries,
        "min_ms": round(min(samples_ns) / 1_000_000, 3),
        "mean_ms": round(sum(samples_ns) / len(samples_ns) / 1_000_000, 3),
        "p50_ms": round(percentile(samples_ns, 0.50) / 1_000_000, 3),
        "p95_ms": round(percentile(samples_ns, 0.95) / 1_000_000, 3),
        "max_ms": round(max(samples_ns) / 1_000_000, 3),
    }


def run(
    *, sizes: tuple[int, ...], dimension: int, queries: int, warmup: int, top_k: int
) -> dict[str, object]:
    rows: list[dict[str, float | int]] = []
    for chunk_count in sizes:
        if top_k > chunk_count:
            raise ValueError("top-k cannot exceed the smallest requested index size")
        index = build_synthetic_index(chunk_count, dimension)
        try:
            row = benchmark_index(index, queries=queries, warmup=warmup, top_k=top_k)
            row.update({"chunk_count": chunk_count, "dimension": dimension, "top_k": top_k})
            rows.append(row)
        finally:
            index.close()
    return {
        "schema_version": 1,
        "scope": "exact LocalVectorIndex.search only; excludes embedding, persistence, BM25, reranking, network, and concurrency",
        "results": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=parse_sizes, default=(100, 1_000, 5_000))
    parser.add_argument("--dimension", type=_positive_int, default=384)
    parser.add_argument("--queries", type=_positive_int, default=30)
    parser.add_argument("--warmup", type=_positive_int, default=5)
    parser.add_argument("--top-k", type=_positive_int, default=5)
    parser.add_argument("--json-output", type=Path)
    arguments = parser.parse_args(argv)

    try:
        report = run(
            sizes=arguments.sizes,
            dimension=arguments.dimension,
            queries=arguments.queries,
            warmup=arguments.warmup,
            top_k=arguments.top_k,
        )
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if arguments.json_output:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
