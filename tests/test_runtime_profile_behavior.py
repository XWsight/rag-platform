from __future__ import annotations

import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from rag_system.application import UploadDocument
from rag_system.bootstrap import LocalDurableRuntimeProfile, build_service_from_settings
from rag_system.config import Settings
from rag_system.domain import AnswerRequest, IndexRef, SearchHit
from rag_system.job_contracts import JobStatus
from rag_system.platform import RagPlatform
from rag_system.runtime_profile import RuntimeComponents
from rag_system.tenancy import Principal, TenantId


class _Index:
    def __init__(self, index_id, chunks) -> None:
        self.index_ref = IndexRef(index_id, len({item.document_id for item in chunks}), len(chunks), time.time())
        self.chunks = tuple(chunks)

    def search(self, _query, *, top_k):
        return tuple(SearchHit(chunk, 0.95 - index * 0.01) for index, chunk in enumerate(self.chunks[:top_k]))

    def close(self) -> None:
        return None

    def delete(self) -> None:
        return None


class _Repository:
    def __init__(self) -> None:
        self.indexes = {}

    def build(self, index_id, chunks):
        index = _Index(index_id, chunks)
        self.indexes[index_id] = index
        return index

    def delete(self, index_id) -> bool:
        return self.indexes.pop(index_id, None) is not None

    def healthcheck(self) -> bool:
        return True


class _ContractProfile(LocalDurableRuntimeProfile):
    """The default durable components with a deterministic test vector adapter."""

    def build_components(self, settings, *, provider_factory=None) -> RuntimeComponents:
        components = super().build_components(settings, provider_factory=provider_factory)
        components.service.close()
        service = build_service_from_settings(settings, index_repository=_Repository())
        return RuntimeComponents(
            service=service,
            catalog=components.catalog,
            file_store=components.file_store,
            jobs=components.jobs,
            idempotency=components.idempotency,
        )


class RuntimeProfileBehaviorTests(unittest.TestCase):
    def test_profile_preserves_create_answer_restart_and_delete_semantics(self) -> None:
        principal = Principal("contract", TenantId("contract"), frozenset({"reader", "writer"}))
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(
                Settings(),
                persist_data=True,
                storage_root=Path(directory),
                job_workers=1,
                max_jobs=8,
                max_jobs_per_tenant=8,
            ).validate()
            profile = _ContractProfile()
            first = profile.build_components(settings)
            platform = self._platform(settings, first)
            try:
                submission = platform.create_knowledge_base(
                    principal,
                    display_name="contract",
                    documents=(UploadDocument("evidence.md", b"contract evidence"),),
                    idempotency_key="runtime-profile-contract",
                )
                self._wait_for_job(platform, principal, submission.job_id)
                record = platform.get_knowledge_base(principal, submission.knowledge_base.resource_id)
                self.assertEqual(record.status.value, "ready")
            finally:
                platform.close()

            restarted = profile.build_components(settings)
            recovered = self._platform(settings, restarted)
            try:
                self.assertEqual(recovered.recover_incomplete((principal,)), 0)
                answer = recovered.answer(
                    principal,
                    record.resource_id,
                    AnswerRequest("contract evidence", "contract-session"),
                )
                self.assertEqual(answer.decision.route.value, "retrieval_only")
                self.assertTrue(answer.citations)
                self.assertTrue(recovered.clear_session(principal, record.resource_id, "contract-session"))
                self.assertTrue(recovered.delete_knowledge_base(principal, record.resource_id))
                self.assertEqual(recovered.list_knowledge_bases(principal), ())
            finally:
                recovered.close()

    @staticmethod
    def _platform(settings, components) -> RagPlatform:
        return RagPlatform(
            settings=settings,
            service=components.service,
            catalog=components.catalog,
            file_store=components.file_store,
            jobs=components.jobs,
            idempotency=components.idempotency,
        )

    def _wait_for_job(self, platform, principal, job_id) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            snapshot = platform.get_job(principal, job_id)
            if snapshot.status.terminal:
                self.assertEqual(snapshot.status, JobStatus.SUCCEEDED)
                return
            time.sleep(0.01)
        self.fail("runtime profile indexing job did not finish")


if __name__ == "__main__":
    unittest.main()
