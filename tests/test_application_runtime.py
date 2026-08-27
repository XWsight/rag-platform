from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rag_system.application_contracts import AnswerPolicy, KnowledgeChatConfiguration, SessionPolicy
from rag_system.application_runtime import (
    ApplicationBoundResourceUnavailableError,
    ApplicationNotPublishedError,
    KnowledgeApplicationRuntime,
)
from rag_system.application_service import ApplicationService
from rag_system.application_store import ApplicationStore
from rag_system.domain import AnswerRequest, AnswerResult, Route, RouteDecision
from rag_system.knowledge_base_contracts import KnowledgeBaseStatus
from rag_system.tenancy import Principal, TenantId


KB_ONE = "kb_12345678901234567890123456789012"
KB_TWO = "kb_22345678901234567890123456789012"


class KnowledgeBaseStub:
    def get(self, principal: Principal, resource_id: str) -> SimpleNamespace:
        if resource_id not in {KB_ONE, KB_TWO}:
            raise LookupError(resource_id)
        return SimpleNamespace(status=KnowledgeBaseStatus.READY)


class RagStub:
    def __init__(self) -> None:
        self.requests: list[tuple[str, AnswerRequest]] = []
        self.cleared: list[tuple[str, str]] = []

    def get_knowledge_base(self, principal: Principal, resource_id: str) -> SimpleNamespace:
        return KnowledgeBaseStub().get(principal, resource_id)

    def answer(self, principal: Principal, resource_id: str, request: AnswerRequest) -> AnswerResult:
        self.requests.append((resource_id, request))
        return AnswerResult("grounded answer", RouteDecision(Route.LOCAL, 1.0, "local"), trace_id="trace")

    def clear_session(self, principal: Principal, resource_id: str, session_id: str) -> bool:
        self.cleared.append((resource_id, session_id))
        return True


class KnowledgeApplicationRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.principal = Principal(
            "operator-user", TenantId("tenant-a"), frozenset({"writer", "operator"})
        )
        self.store = ApplicationStore(Path(self.directory.name, "applications.sqlite3"))
        self.service = ApplicationService(self.store, KnowledgeBaseStub(), clock=lambda: 10.0)
        self.rag = RagStub()
        self.runtime = KnowledgeApplicationRuntime(self.store, self.rag)  # type: ignore[arg-type]

    def create_application(self):
        project = self.service.create_project(self.principal, "Support")
        return self.service.create_knowledge_application(
            self.principal, project.project_id, "Support assistant"
        )

    def test_resolves_published_revision_and_maps_policy_without_leaking_session_id(self) -> None:
        application = self.create_application()
        revision = self.service.create_knowledge_revision(
            self.principal,
            application.application_id,
            KnowledgeChatConfiguration(
                knowledge_base_ids=(KB_ONE,),
                answer_policy=AnswerPolicy(allow_cloud=True, allow_web=True, allow_research=True),
                session_policy=SessionPolicy(enabled=False),
            ),
            change_summary="Initial release",
        )
        self.service.publish(self.principal, application.application_id, revision.revision_id)

        answer = self.runtime.answer(
            self.principal, application.application_id, question="What is covered?", session_id="browser-1"
        )

        self.assertEqual(answer.application_id, application.application_id)
        self.assertEqual(answer.revision_id, revision.revision_id)
        self.assertEqual(answer.result.trace_id, "trace")
        resource_id, request = self.rag.requests[0]
        self.assertEqual(resource_id, KB_ONE)
        self.assertTrue(request.allow_cloud)
        self.assertTrue(request.allow_web)
        self.assertTrue(request.deep_research)
        self.assertNotIn("browser-1", request.session_id)
        self.assertEqual(self.rag.cleared, [(KB_ONE, request.session_id)])

    def test_refuses_unpublished_and_multiple_bound_knowledge_bases(self) -> None:
        application = self.create_application()
        with self.assertRaises(ApplicationNotPublishedError):
            self.runtime.answer(
                self.principal, application.application_id, question="Question", session_id="browser-1"
            )
        revision = self.service.create_knowledge_revision(
            self.principal,
            application.application_id,
            KnowledgeChatConfiguration(knowledge_base_ids=(KB_ONE, KB_TWO)),
            change_summary="Multiple corpora",
        )
        self.service.publish(self.principal, application.application_id, revision.revision_id)

        with self.assertRaises(ApplicationBoundResourceUnavailableError):
            self.runtime.answer(
                self.principal, application.application_id, question="Question", session_id="browser-1"
            )


if __name__ == "__main__":
    unittest.main()
