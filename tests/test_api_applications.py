from __future__ import annotations

import tempfile
import unittest
import logging
from pathlib import Path

from fastapi.testclient import TestClient

from rag_system.api import create_app
from rag_system.application_runtime import KnowledgeApplicationRuntime
from rag_system.application_service import ApplicationService
from rag_system.application_store import ApplicationStore
from rag_system.observability import JsonEventLogger
from rag_system.rate_limit import TokenBucketRateLimiter
from rag_system.tenancy import ApiKeyAuthenticator, Principal, TenantId
from tests.test_api import ALL_ROLES_KEY, KNOWLEDGE_BASE_ID, FakePlatform


class KnowledgeBaseRepositoryStub:
    def __init__(self, platform: FakePlatform) -> None:
        self._platform = platform

    def get(self, principal: Principal, resource_id: str):
        return self._platform.get_knowledge_base(principal, resource_id)


class ApplicationApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.platform = FakePlatform()
        self.principal = Principal(
            "application-operator",
            TenantId("tenant-one"),
            frozenset({"reader", "writer", "operator"}),
        )
        self.store = ApplicationStore(Path(self.directory.name, "applications.sqlite3"))
        self.service = ApplicationService(
            self.store, KnowledgeBaseRepositoryStub(self.platform)  # type: ignore[arg-type]
        )
        self.runtime = KnowledgeApplicationRuntime(self.store, self.platform)
        app = create_app(
            platform=self.platform,
            authenticator=ApiKeyAuthenticator.from_mapping({ALL_ROLES_KEY: self.principal}),
            rate_limiter=TokenBucketRateLimiter(rate_per_second=100, capacity=100),
            logger=JsonEventLogger(logging.getLogger(f"application-api-{id(self)}")),
            application_service=self.service,
            application_runtime=self.runtime,
        )
        self.context = TestClient(app, raise_server_exceptions=False)
        self.client = self.context.__enter__()

    def tearDown(self) -> None:
        self.context.__exit__(None, None, None)

    @property
    def headers(self) -> dict[str, str]:
        return {"X-API-Key": ALL_ROLES_KEY}

    def test_create_publish_answer_and_rollback_are_version_traceable(self) -> None:
        project = self.client.post(
            "/v1/projects", headers=self.headers,
            json={"display_name": "Support", "description": "Customer support"},
        )
        self.assertEqual(project.status_code, 200)
        application = self.client.post(
            "/v1/applications", headers=self.headers,
            json={
                "project_id": project.json()["id"], "display_name": "Support assistant",
                "application_kind": "knowledge_chat",
            },
        )
        self.assertEqual(application.status_code, 200)
        application_id = application.json()["id"]

        first = self._create_revision(application_id, "Initial release", allow_cloud=False)
        second = self._create_revision(application_id, "Cloud enabled", allow_cloud=True)
        published = self.client.post(
            f"/v1/applications/{application_id}/deployments", headers=self.headers,
            json={"revision_id": second, "expected_active_revision_id": None},
        )
        self.assertEqual(published.status_code, 200)
        answer = self.client.post(
            f"/v1/apps/{application_id}/answer", headers=self.headers,
            json={"question": "What is RAG?", "session_id": "browser-1"},
        )
        self.assertEqual(answer.status_code, 200)
        self.assertEqual(answer.json()["application_id"], application_id)
        self.assertEqual(answer.json()["revision_id"], second)
        self.assertTrue(answer.json()["trace_id"])
        self.assertTrue(self.platform.last_answer_request.allow_cloud)

        rolled_back = self.client.post(
            f"/v1/applications/{application_id}/deployments", headers=self.headers,
            json={"revision_id": first, "expected_active_revision_id": second},
        )
        self.assertEqual(rolled_back.status_code, 200)
        active = self.client.get(f"/v1/applications/{application_id}", headers=self.headers)
        self.assertEqual(active.json()["active_revision_id"], first)
        self.assertEqual(
            self.client.get(
                f"/v1/applications/{application_id}/revisions", headers=self.headers
            ).json()["count"],
            2,
        )
        archived = self.client.delete(
            f"/v1/applications/{application_id}", headers=self.headers
        )
        self.assertEqual(archived.status_code, 200)
        self.assertEqual(archived.json()["status"], "archived")
        refused = self.client.post(
            f"/v1/apps/{application_id}/answer", headers=self.headers,
            json={"question": "What is RAG?", "session_id": "browser-2"},
        )
        self.assertEqual(refused.status_code, 409)
        self.assertEqual(refused.json()["error"]["code"], "application_not_published")

    def test_openapi_and_strict_payloads_expose_the_platform_contract(self) -> None:
        document = self.client.get("/openapi.json").json()
        self.assertIn("/v1/apps/{application_id}/answer", document["paths"])
        self.assertIn("/v1/applications/{application_id}/deployments", document["paths"])
        response = self.client.post(
            "/v1/projects", headers=self.headers,
            json={"display_name": "Support", "unexpected": True},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "invalid_request")

    def test_draft_update_snapshot_and_publish_conflict_are_explicit(self) -> None:
        project = self.client.post(
            "/v1/projects", headers=self.headers, json={"display_name": "Support"}
        ).json()
        application = self.client.post(
            "/v1/applications",
            headers=self.headers,
            json={"project_id": project["id"], "display_name": "Support", "application_kind": "knowledge_chat"},
        ).json()
        draft_path = f"/v1/applications/{application['id']}/draft"
        self.assertEqual(self.client.get(draft_path, headers=self.headers).json()["version"], 0)
        updated = self.client.put(
            draft_path,
            headers=self.headers,
            json={
                "expected_version": 0,
                "knowledge_base_ids": [KNOWLEDGE_BASE_ID],
                "retrieval_profile": "focused",
                "answer_policy": {"require_citations": False, "allow_cloud": True, "allow_web": False, "allow_research": False},
                "session_policy": {"enabled": True, "ttl_seconds": 3600},
                "change_summary": "Prepared release",
            },
        )
        self.assertEqual(updated.status_code, 200, updated.text)
        revision = self.client.post(
            f"{draft_path}/revisions", headers=self.headers, json={"expected_version": 1}
        )
        self.assertEqual(revision.status_code, 200, revision.text)
        first = self.client.post(
            f"/v1/applications/{application['id']}/deployments",
            headers=self.headers,
            json={"revision_id": revision.json()["id"], "expected_active_revision_id": None},
        )
        self.assertEqual(first.status_code, 200)
        stale = self.client.post(
            f"/v1/applications/{application['id']}/deployments",
            headers=self.headers,
            json={"revision_id": revision.json()["id"], "expected_active_revision_id": None},
        )
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.json()["error"]["code"], "application_conflict")

    def _create_revision(self, application_id: str, summary: str, *, allow_cloud: bool) -> str:
        response = self.client.post(
            f"/v1/applications/{application_id}/revisions", headers=self.headers,
            json={
                "knowledge_base_ids": [KNOWLEDGE_BASE_ID],
                "answer_policy": {
                    "require_citations": True, "allow_cloud": allow_cloud,
                    "allow_web": False, "allow_research": False,
                },
                "session_policy": {"enabled": True, "ttl_seconds": 3600},
                "change_summary": summary,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return str(response.json()["id"])


if __name__ == "__main__":
    unittest.main()
