"""Packaged ASGI entry point for the durable RAG Platform API."""

from __future__ import annotations

from rag_system.api import create_app
from rag_system.runtime_bootstrap import build_production_runtime


runtime = build_production_runtime()
app = create_app(
    platform=runtime.platform,
    authenticator=runtime.authenticator,
    rate_limiter=runtime.rate_limiter,
    logger=runtime.event_logger,
    readiness=runtime.ready,
    shutdown=runtime.close,
    application_service=runtime.application_service,
    application_runtime=runtime.application_runtime,
)
