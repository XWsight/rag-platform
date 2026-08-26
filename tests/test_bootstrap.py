from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from rag_system.bootstrap import StorageRootLease, build_service_from_settings, parse_api_credentials
from rag_system.config import SecretValue, Settings
from rag_system.domain import GeneratedAnswer, WebSearchResult
from rag_system.provider_factory import ProviderBundle


class _TestChat:
    available = True

    def answer(self, question: str, evidence: object) -> GeneratedAnswer:
        return GeneratedAnswer((), insufficient=True)

    def plan_queries(self, question: str, *, max_queries: int) -> tuple[str, ...]:
        return ()


class _TestWebSearch:
    available = True

    def search(self, query: str, *, count: int) -> tuple[WebSearchResult, ...]:
        return ()


class _TestProviderFactory:
    def __init__(self) -> None:
        self.settings: Settings | None = None
        self.chat_model = _TestChat()
        self.web_search = _TestWebSearch()

    def create(self, settings: Settings) -> ProviderBundle:
        self.settings = settings
        return ProviderBundle(
            chat_model=self.chat_model,
            web_search=self.web_search,
            query_planner=self.chat_model,
        )


class BootstrapTests(unittest.TestCase):
    def test_credentials_are_strict_and_raw_keys_are_not_retained_in_repr(self) -> None:
        raw_key = "0123456789abcdef0123456789abcdef"
        encoded = SecretValue(
            '{"'
            + raw_key
            + '":{"subject":"operator-1","tenant_id":"tenant-a",'
            '"roles":["reader","writer","operator"]}}'
        )
        authenticator, principals = parse_api_credentials(encoded)

        self.assertEqual(authenticator.authenticate(raw_key), principals[0])
        self.assertNotIn(raw_key, repr(authenticator))
        self.assertEqual(principals[0].tenant_id.value, "tenant-a")

    def test_duplicate_json_keys_and_unknown_fields_are_rejected(self) -> None:
        key = "0123456789abcdef"
        duplicate = SecretValue(
            f'{{"{key}":{{"subject":"user-1","tenant_id":"tenant-a",'
            f'"roles":["reader"]}},"{key}":{{"subject":"user-2",'
            '"tenant_id":"tenant-b","roles":["reader"]}}}'
        )
        unknown = SecretValue(
            f'{{"{key}":{{"subject":"user-1","tenant_id":"tenant-a",'
            '"roles":["reader"],"extra":true}}}'
        )

        with self.assertRaisesRegex(ValueError, "RAG_API_KEYS_JSON"):
            parse_api_credentials(duplicate)
        with self.assertRaisesRegex(ValueError, "RAG_API_KEYS_JSON"):
            parse_api_credentials(unknown)

    def test_storage_root_lease_rejects_a_second_process_slot_and_releases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = StorageRootLease.acquire(root)
            try:
                with self.assertRaisesRegex(RuntimeError, "already in use"):
                    StorageRootLease.acquire(root)
            finally:
                first.close()

            second = StorageRootLease.acquire(root)
            second.close()

    def test_service_bootstrap_accepts_an_explicit_provider_factory(self) -> None:
        factory = _TestProviderFactory()

        service = build_service_from_settings(Settings(), provider_factory=factory)

        self.assertIsNotNone(factory.settings)
        self.assertIs(service.chat_model, factory.chat_model)
        self.assertIs(service.web_search, factory.web_search)
        self.assertIs(service.query_planner, factory.chat_model)


if __name__ == "__main__":
    unittest.main()
