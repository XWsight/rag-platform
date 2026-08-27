"""Compatibility ASGI entry point; new deployments use ``rag_system.asgi``."""

from rag_system.asgi import app


__all__ = ["app"]
