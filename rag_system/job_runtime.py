"""Private in-memory state and bounded retention for one job executor."""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from rag_system.job_contracts import JobCapacityError, JobId, JobStatus


@dataclass(slots=True)
class JobRecord:
    """Mutable executor-owned state; never expose this outside the executor lock."""

    job_id: JobId
    tenant_id: str
    idempotency_key: str
    status: JobStatus
    created_at: float
    updated_at: float
    last_accessed_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result_json: str | None = None
    error_code: str = ""
    error_message: str = ""
    cancellation: threading.Event = field(default_factory=threading.Event)
    future: Future[Any] | None = None


class JobRetention:
    """Maintain LRU retention and idempotency indexes under an owner-held lock."""

    def __init__(self, *, max_jobs: int, max_jobs_per_tenant: int, ttl_seconds: float) -> None:
        self._max_jobs = max_jobs
        self._max_jobs_per_tenant = max_jobs_per_tenant
        self._ttl_seconds = ttl_seconds
        self._records: OrderedDict[str, JobRecord] = OrderedDict()
        self._idempotency: dict[tuple[str, str], str] = {}

    def get(self, job_id: str) -> JobRecord | None:
        return self._records.get(job_id)

    def contains(self, job_id: str) -> bool:
        return job_id in self._records

    def values(self) -> tuple[JobRecord, ...]:
        return tuple(self._records.values())

    def add(self, record: JobRecord) -> None:
        self._records[record.job_id.value] = record
        self._idempotency[(record.tenant_id, record.idempotency_key)] = record.job_id.value

    def find_idempotent(self, tenant_id: str, key: str) -> JobRecord | None:
        identity = (tenant_id, key)
        job_id = self._idempotency.get(identity)
        if job_id is None:
            return None
        record = self._records.get(job_id)
        if record is None:
            self._idempotency.pop(identity, None)
        return record

    def touch(self, record: JobRecord, now: float) -> None:
        record.last_accessed_at = now
        self._records.move_to_end(record.job_id.value)

    def cleanup_expired(self, now: float) -> int:
        expired = tuple(
            job_id
            for job_id, record in self._records.items()
            if record.status.terminal and now - record.last_accessed_at >= self._ttl_seconds
        )
        for job_id in expired:
            self.remove(job_id)
        return len(expired)

    def make_capacity(self) -> None:
        while len(self._records) >= self._max_jobs:
            terminal_id = next(
                (job_id for job_id, record in self._records.items() if record.status.terminal),
                None,
            )
            if terminal_id is None:
                raise JobCapacityError("job capacity reached")
            self.remove(terminal_id)

    def make_tenant_capacity(self, tenant_id: str) -> None:
        tenant_records = [
            (job_id, record)
            for job_id, record in self._records.items()
            if record.tenant_id == tenant_id
        ]
        while len(tenant_records) >= self._max_jobs_per_tenant:
            terminal = next(
                ((job_id, record) for job_id, record in tenant_records if record.status.terminal),
                None,
            )
            if terminal is None:
                raise JobCapacityError("tenant job capacity reached")
            self.remove(terminal[0])
            tenant_records.remove(terminal)

    def remove(self, job_id: str) -> JobRecord | None:
        record = self._records.pop(job_id, None)
        if record is None:
            return None
        identity = (record.tenant_id, record.idempotency_key)
        if self._idempotency.get(identity) == job_id:
            self._idempotency.pop(identity, None)
        return record
