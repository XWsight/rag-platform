"""Command-line launcher for the single-worker RAG Platform API."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn


def build_parser() -> argparse.ArgumentParser:
    """Build a deliberately small, single-node-safe server command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: loopback).")
    parser.add_argument("--port", type=_port, default=8000, help="TCP port (default: 8000).")
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug"),
        default="info",
        help="Uvicorn log level (default: info).",
    )
    parser.add_argument(
        "--access-log",
        action="store_true",
        help="Enable Uvicorn access logs; disabled by default to minimize request metadata retention.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the packaged ASGI application with exactly one worker."""

    arguments = build_parser().parse_args(argv)
    uvicorn.run(
        "rag_system.asgi:app",
        host=arguments.host,
        port=arguments.port,
        log_level=arguments.log_level,
        access_log=arguments.access_log,
        workers=1,
        timeout_graceful_shutdown=30,
    )
    return 0


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


if __name__ == "__main__":
    raise SystemExit(main())
