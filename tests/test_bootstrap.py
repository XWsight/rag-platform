from __future__ import annotations

import unittest
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rag_system.bootstrap import (
    LocalDurableRuntimeProfile,
    StorageRootLease,
    build_production_runtime,
    build_service_from_settings,
    parse_api_credentials,
)
from rag_system.config import SecretValue, Settings
from rag_system.domain import GeneratedAnswer, WebSearchResult
from rag_system.health import HealthProbe
from rag_system.provider_factory import ProviderBundle
from rag_system.runtime_profile import RuntimeComponents


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


class _RuntimeIndexManager:
    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True

    def healthcheck(self) -> bool:
        return True


class _RuntimeService:
    def __init__(self) -> None:
        self.index_manager = _RuntimeIndexManager()
        self.closed = False
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class _RuntimeJobs:
    def __init__(self, *, fail_shutdown: bool = False) -> None:
        self.shutdown_calls = 0
        self.fail_shutdown = fail_shutdown

    def shutdown(self, *, wait: bool = True, cancel_pending: bool = True) -> None:
        self.shutdown_calls += 1
        if self.fail_shutdown:
            raise RuntimeError("job shutdown failure")

    def healthcheck(self) -> bool:
        return True


class _RuntimeCatalog:
    def __init__(self, *, fail_listing: bool = False) -> None:
        self.fail_listing = fail_listing

    def list(self, *_args, **_kwargs) -> tuple[object, ...]:
        if self.fail_listing:
            raise RuntimeError("catalog startup failure")
        return ()


class _RuntimeFileStore:
    def healthcheck(self) -> bool:
        return True


class _RuntimeIdempotency:
    pass


class _TestRuntimeProfile:
    def __init__(self, components: RuntimeComponents) -> None:
        self.components = components
        self.calls = 0

    def build_components(self, _settings, *, provider_factory=None) -> RuntimeComponents:
        self.calls += 1
        return self.components

    def readiness_probes(self, _components, _principals) -> tuple[HealthProbe, ...]:
        return (HealthProbe("profile", lambda: True),)


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

    def test_storage_root_lease_rejects_a_non_file_lease_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".rag-studio.instance").mkdir()

            with self.assertRaisesRegex(RuntimeError, "lease path is unsafe"):
                StorageRootLease.acquire(root)

    def test_service_bootstrap_accepts_an_explicit_provider_factory(self) -> None:
        factory = _TestProviderFactory()

        service = build_service_from_settings(Settings(), provider_factory=factory)

        self.assertIsNotNone(factory.settings)
        self.assertIs(service.chat_model, factory.chat_model)
        self.assertIs(service.web_search, factory.web_search)
        self.assertIs(service.query_planner, factory.chat_model)

    def test_default_profile_cleans_a_service_when_durable_job_recovery_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(), storage_root=Path(directory))
            service = _RuntimeService()
            with (
                patch("rag_system.bootstrap.build_service_from_settings", return_value=service),
                patch(
                    "rag_system.bootstrap.SqliteJobSnapshotStore.recover_interrupted",
                    side_effect=RuntimeError("durable job recovery failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "durable job recovery failure"):
                    LocalDurableRuntimeProfile().build_components(settings)

            self.assertEqual(service.index_manager.close_calls, 1)
            self.assertEqual(service.close_calls, 1)

    def test_production_runtime_accepts_a_replaceable_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(
                Settings(),
                persist_data=True,
                storage_root=Path(directory),
                api_keys_json=SecretValue(
                    '{"0123456789abcdef":{"subject":"operator",'
                    '"tenant_id":"tenant-a","roles":["operator"]}}'
                ),
            )
            service = _RuntimeService()
            jobs = _RuntimeJobs()
            profile = _TestRuntimeProfile(
                RuntimeComponents(
                    service=service,
                    catalog=_RuntimeCatalog(),
                    file_store=_RuntimeFileStore(),
                    jobs=jobs,
                    idempotency=_RuntimeIdempotency(),
                )
            )
            with patch("rag_system.bootstrap.load_settings", return_value=settings):
                runtime = build_production_runtime(runtime_profile=profile)
            try:
                self.assertEqual(profile.calls, 1)
                self.assertTrue(runtime.ready())
            finally:
                runtime.close()
                runtime.close()

            self.assertEqual(jobs.shutdown_calls, 1)
            self.assertEqual(service.index_manager.close_calls, 1)
            self.assertEqual(service.close_calls, 1)
            self.assertTrue(service.index_manager.closed)
            self.assertTrue(service.closed)

    def test_startup_failure_closes_components_owned_by_a_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(
                Settings(),
                persist_data=True,
                storage_root=Path(directory),
                api_keys_json=SecretValue(
                    '{"0123456789abcdef":{"subject":"operator",'
                    '"tenant_id":"tenant-a","roles":["operator"]}}'
                ),
            )
            service = _RuntimeService()
            jobs = _RuntimeJobs()
            profile = _TestRuntimeProfile(
                RuntimeComponents(
                    service=service,
                    catalog=_RuntimeCatalog(fail_listing=True),
                    file_store=_RuntimeFileStore(),
                    jobs=jobs,
                    idempotency=_RuntimeIdempotency(),
                )
            )
            with patch("rag_system.bootstrap.load_settings", return_value=settings):
                with self.assertRaisesRegex(RuntimeError, "catalog startup failure"):
                    build_production_runtime(runtime_profile=profile)

            self.assertEqual(jobs.shutdown_calls, 1)
            self.assertTrue(service.index_manager.closed)
            self.assertTrue(service.closed)
            recovered_lease = StorageRootLease.acquire(Path(directory))
            recovered_lease.close()

    def test_runtime_close_finishes_cleanup_after_a_shutdown_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(
                Settings(),
                persist_data=True,
                storage_root=Path(directory),
                api_keys_json=SecretValue(
                    '{"0123456789abcdef":{"subject":"operator",'
                    '"tenant_id":"tenant-a","roles":["operator"]}}'
                ),
            )
            service = _RuntimeService()
            jobs = _RuntimeJobs(fail_shutdown=True)
            profile = _TestRuntimeProfile(
                RuntimeComponents(
                    service=service,
                    catalog=_RuntimeCatalog(),
                    file_store=_RuntimeFileStore(),
                    jobs=jobs,
                    idempotency=_RuntimeIdempotency(),
                )
            )
            with patch("rag_system.bootstrap.load_settings", return_value=settings):
                runtime = build_production_runtime(runtime_profile=profile)

            with self.assertRaisesRegex(RuntimeError, "job shutdown failure"):
                runtime.close()
            runtime.close()

            self.assertEqual(jobs.shutdown_calls, 1)
            self.assertEqual(service.index_manager.close_calls, 1)
            self.assertEqual(service.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
