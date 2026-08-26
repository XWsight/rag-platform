"""Provider-neutral assembly contracts for cloud-backed RAG capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

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

    def __post_init__(self) -> None:
        """Reject incomplete adapters while the composition root is starting."""

        if not isinstance(self.chat_model, ChatModel):
            raise TypeError("provider chat_model does not implement ChatModel")
        if not isinstance(self.web_search, WebSearchProvider):
            raise TypeError("provider web_search does not implement WebSearchProvider")
        if self.query_planner is not None and not isinstance(self.query_planner, QueryPlanner):
            raise TypeError("provider query_planner does not implement QueryPlanner")


@runtime_checkable
class ProviderFactory(Protocol):
    """Build the cloud provider adapters used by a service composition root."""

    def create(self, settings: Settings) -> ProviderBundle: ...


def create_provider_bundle(factory: ProviderFactory, settings: Settings) -> ProviderBundle:
    """Create a verified provider bundle without allowing dynamic code loading.

    This deliberately validates only stable port shape. Provider-specific wire
    behavior remains covered by that adapter's isolated tests, while this
    boundary makes an incomplete factory fail at startup rather than during a
    user request.
    """

    if not isinstance(factory, ProviderFactory):
        raise TypeError("provider_factory must define create(Settings) -> ProviderBundle")
    providers = factory.create(settings)
    if not isinstance(providers, ProviderBundle):
        raise TypeError("provider_factory must return a ProviderBundle")
    return providers


__all__ = ["ProviderBundle", "ProviderFactory", "create_provider_bundle"]
