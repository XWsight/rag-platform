from __future__ import annotations

import unittest

from rag_system.job_contracts import JobCapacityError, JobId, JobStatus
from rag_system.job_runtime import JobRecord, JobRetention


def _record(
    value: str,
    *,
    tenant: str = "tenant-a",
    key: str = "request",
    status: JobStatus = JobStatus.QUEUED,
    timestamp: float = 0.0,
) -> JobRecord:
    return JobRecord(
        job_id=JobId(value),
        tenant_id=tenant,
        idempotency_key=key,
        status=status,
        created_at=timestamp,
        updated_at=timestamp,
        last_accessed_at=timestamp,
        finished_at=timestamp if status.terminal else None,
    )


class JobRetentionTests(unittest.TestCase):
    def test_retention_never_evicts_active_work_and_keeps_idempotency_scoped(self) -> None:
        retention = JobRetention(max_jobs=2, max_jobs_per_tenant=1, ttl_seconds=10.0)
        active = _record("active", key="same")
        terminal = _record("terminal", key="done", status=JobStatus.SUCCEEDED)
        retention.add(active)
        retention.add(terminal)

        self.assertIs(retention.find_idempotent("tenant-a", "same"), active)
        self.assertIsNone(retention.find_idempotent("tenant-b", "same"))
        retention.make_capacity()
        self.assertIs(retention.get("active"), active)
        self.assertIsNone(retention.get("terminal"))
        with self.assertRaises(JobCapacityError):
            retention.make_tenant_capacity("tenant-a")

    def test_ttl_cleanup_removes_terminal_records_and_their_idempotency_mapping(self) -> None:
        retention = JobRetention(max_jobs=2, max_jobs_per_tenant=2, ttl_seconds=10.0)
        terminal = _record("done", key="request", status=JobStatus.SUCCEEDED, timestamp=1.0)
        retention.add(terminal)

        self.assertEqual(retention.cleanup_expired(11.0), 1)
        self.assertIsNone(retention.get("done"))
        self.assertIsNone(retention.find_idempotent("tenant-a", "request"))


if __name__ == "__main__":
    unittest.main()
