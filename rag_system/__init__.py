"""Core package for RAG Platform."""

from rag_system.config import Settings
from rag_system.domain import AnswerRequest, AnswerResult, Route

__all__ = ["AnswerRequest", "AnswerResult", "Route", "Settings"]
# Keep this aligned with the PEP 621 version in ``pyproject.toml``.  The API
# exposes this value as well, so a pre-release can never masquerade as a
# stable build.
__version__ = "2.0.0.dev0"
