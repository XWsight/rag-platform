from __future__ import annotations

import unittest

from rag_system.config import Settings
from rag_system.health import HealthProbe
from rag_system.runtime_profile import (
    RuntimeComponents,
    RuntimeProfileConformanceError,
    verify_runtime_profile,
)
from rag_system.tenancy import Principal, TenantId


class _IndexManager:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _Service:
    def __init__(self) -> None:
        self.index_manager = _IndexManager()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _Jobs:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self, *, wait: bool = True, cancel_pending: bool = True) -> None:
        self.shutdown_calls += 1


class _Profile:
    def __init__(self, *, ready: bool = True) -> None:
        self.service = _Service()
        self.jobs = _Jobs()
        self.ready = ready

    def build_components(self, _settings, *, provider_factory=None) -> RuntimeComponents:
        return RuntimeComponents(
            service=self.service,
            catalog=object(),
            file_store=object(),
            jobs=self.jobs,
            idempotency=object(),
        )

    def readiness_probes(self, _components, _principals) -> tuple[HealthProbe, ...]:
        return (HealthProbe("profile", lambda: self.ready),)


class RuntimeProfileConformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.principal = Principal(
            subject="operator",
            tenant_id=TenantId("tenant-a"),
            roles=frozenset({"operator"}),
        )

    def test_verification_reports_ready_probe_and_releases_components(self) -> None:
        profile = _Profile()

        result = verify_runtime_profile(profile, Settings(), (self.principal,))

        self.assertEqual(result.probe_names, ("profile",))
        self.assertEqual(profile.jobs.shutdown_calls, 1)
        self.assertEqual(profile.service.index_manager.close_calls, 1)
        self.assertEqual(profile.service.close_calls, 1)

    def test_verification_rejects_unready_profile_and_still_releases_components(self) -> None:
        profile = _Profile(ready=False)

        with self.assertRaisesRegex(RuntimeProfileConformanceError, "unavailable: profile"):
            verify_runtime_profile(profile, Settings(), (self.principal,))

        self.assertEqual(profile.jobs.shutdown_calls, 1)
        self.assertEqual(profile.service.index_manager.close_calls, 1)
        self.assertEqual(profile.service.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
