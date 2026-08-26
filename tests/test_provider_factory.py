from __future__ import annotations

import unittest

from rag_system.config import Settings
from rag_system.domain import GeneratedAnswer, WebSearchResult
from rag_system.provider_factory import ProviderBundle, create_provider_bundle


class _Chat:
    available = True

    def answer(self, question: str, evidence: object) -> GeneratedAnswer:
        return GeneratedAnswer((), insufficient=True)


class _WebSearch:
    available = True

    def search(self, query: str, *, count: int) -> tuple[WebSearchResult, ...]:
        return ()


class _Planner:
    available = True

    def plan_queries(self, question: str, *, max_queries: int) -> tuple[str, ...]:
        return ()


class _Factory:
    def create(self, settings: Settings) -> ProviderBundle:
        return ProviderBundle(_Chat(), _WebSearch(), _Planner())


class _WrongReturnFactory:
    def create(self, settings: Settings) -> object:
        return object()


class ProviderFactoryTests(unittest.TestCase):
    def test_valid_factory_creates_a_verified_bundle(self) -> None:
        providers = create_provider_bundle(_Factory(), Settings())

        self.assertTrue(providers.chat_model.available)
        self.assertTrue(providers.web_search.available)
        self.assertTrue(providers.query_planner is not None)

    def test_bundle_rejects_adapters_that_do_not_implement_ports(self) -> None:
        with self.assertRaisesRegex(TypeError, "chat_model"):
            ProviderBundle(object(), _WebSearch())
        with self.assertRaisesRegex(TypeError, "web_search"):
            ProviderBundle(_Chat(), object())
        with self.assertRaisesRegex(TypeError, "query_planner"):
            ProviderBundle(_Chat(), _WebSearch(), object())

    def test_factory_must_return_the_stable_bundle_type(self) -> None:
        with self.assertRaisesRegex(TypeError, "ProviderBundle"):
            create_provider_bundle(_WrongReturnFactory(), Settings())  # type: ignore[arg-type]

    def test_factory_must_expose_the_create_contract(self) -> None:
        with self.assertRaisesRegex(TypeError, "create"):
            create_provider_bundle(object(), Settings())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
