from __future__ import annotations

import unittest

from rag_system.runtime_profile import RuntimeComponents


class _IndexManager:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.failure is not None:
            raise self.failure


class _Service:
    def __init__(self, *, index_failure: Exception | None = None, failure: Exception | None = None) -> None:
        self.index_manager = _IndexManager(failure=index_failure)
        self.failure = failure
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.failure is not None:
            raise self.failure


class _Jobs:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.shutdown_calls = 0

    def shutdown(self, *, wait: bool = True, cancel_pending: bool = True) -> None:
        self.shutdown_calls += 1
        if self.failure is not None:
            raise self.failure


class RuntimeComponentsTests(unittest.TestCase):
    @staticmethod
    def _components(service: _Service, jobs: _Jobs) -> RuntimeComponents:
        return RuntimeComponents(
            service=service,
            catalog=object(),
            file_store=object(),
            jobs=jobs,
            idempotency=object(),
        )

    def test_close_releases_every_owned_resource(self) -> None:
        service = _Service()
        jobs = _Jobs()

        self._components(service, jobs).close()

        self.assertEqual(jobs.shutdown_calls, 1)
        self.assertEqual(service.index_manager.close_calls, 1)
        self.assertEqual(service.close_calls, 1)

    def test_close_preserves_first_failure_after_attempting_every_cleanup(self) -> None:
        service = _Service(
            index_failure=RuntimeError("index cleanup failure"),
            failure=RuntimeError("service cleanup failure"),
        )
        jobs = _Jobs(failure=RuntimeError("job cleanup failure"))

        with self.assertRaisesRegex(RuntimeError, "job cleanup failure"):
            self._components(service, jobs).close()

        self.assertEqual(jobs.shutdown_calls, 1)
        self.assertEqual(service.index_manager.close_calls, 1)
        self.assertEqual(service.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
