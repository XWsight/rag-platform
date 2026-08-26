"""Provider-neutral assembly contracts for cloud-backed RAG capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag_system.config import SecretValue, Settings
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


def verify_offline_provider_factory(factory: ProviderFactory) -> None:
    """Exercise a factory's safe no-credential startup contract.

    This intentionally does not call provider operations: adapter-specific
    transport, error-redaction, and answer-protocol tests remain the
    derivative's responsibility.  It does ensure that a factory can assemble
    without credentials, reports strict boolean availability, and releases
    optional client resources idempotently.
    """

    settings = Settings(api_key=SecretValue("")).validate()
    providers = create_provider_bundle(factory, settings)
    adapters = tuple(
        adapter
        for adapter in (providers.chat_model, providers.web_search, providers.query_planner)
        if adapter is not None
    )
    for adapter in adapters:
        available = adapter.available
        if not isinstance(available, bool):
            raise TypeError("provider availability must be a boolean")
        if available:
            raise ValueError("provider must be unavailable when credentials are absent")

    closed: set[int] = set()
    for adapter in adapters:
        if id(adapter) in closed:
            continue
        closed.add(id(adapter))
        close = getattr(adapter, "close", None)
        if close is None:
            continue
        if not callable(close):
            raise TypeError("provider close attribute must be callable")
        close()
        close()


__all__ = [
    "ProviderBundle",
    "ProviderFactory",
    "create_provider_bundle",
    "verify_offline_provider_factory",
]
