"""Provider-neutral assembly contracts for cloud-backed RAG capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag_system.config import Settings
from rag_system.ports import ChatModel, QueryPlanner, WebSearchProvider


@dataclass(frozen=True, slots=True)
class ProviderBundle:
    """Concrete provider adapters assembled for one application runtime.

    The application keeps its trust boundaries in the service layer. Factories
    only choose transports and credentials-backed adapters; they must not
    bypass grounded-answer validation or request-level outbound authorization.
    """

    chat_model: ChatModel
    web_search: WebSearchProvider
    query_planner: QueryPlanner | None = None


class ProviderFactory(Protocol):
    """Build the cloud provider adapters used by a service composition root."""

    def create(self, settings: Settings) -> ProviderBundle: ...


__all__ = ["ProviderBundle", "ProviderFactory"]
