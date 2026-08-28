from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rag_system.application_contracts import (
    ApplicationAuditEventType,
    ApplicationStatus,
    KnowledgeChatConfiguration,
)
from rag_system.application_service import (
    ApplicationAuthorizationError,
    ApplicationResourceUnavailableError,
    ApplicationService,
    ApplicationServiceValidationError,
)
from rag_system.application_store import ApplicationStore
from rag_system.knowledge_base_contracts import KnowledgeBaseStatus
from rag_system.tenancy import Principal, TenantId


KNOWLEDGE_BASE_ID = "kb_12345678901234567890123456789012"


class KnowledgeBaseStub:
    def __init__(self, status: KnowledgeBaseStatus) -> None:
        self.status = status

    def get(self, principal: Principal, resource_id: str) -> SimpleNamespace:
        if resource_id != KNOWLEDGE_BASE_ID:
            raise LookupError(resource_id)
        return SimpleNamespace(status=self.status)


class ApplicationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = ApplicationStore(Path(self.directory.name, "applications.sqlite3"))
        self.writer = Principal(
            "writer-user", TenantId("tenant-a"), frozenset({"writer", "operator"})
        )
        self.reader = Principal("reader-user", TenantId("tenant-a"), frozenset({"reader"}))
        self.knowledge_bases = KnowledgeBaseStub(KnowledgeBaseStatus.READY)
        self.service = ApplicationService(self.store, self.knowledge_bases, clock=lambda: 10.0)

    def test_create_version_publish_and_rollback_keep_revisions_immutable(self) -> None:
        project = self.service.create_project(self.writer, "Support", "Support automation")
        application = self.service.create_knowledge_application(
            self.writer, project.project_id, "Support assistant"
        )
        configuration = KnowledgeChatConfiguration(knowledge_base_ids=(KNOWLEDGE_BASE_ID,))
        first = self.service.create_knowledge_revision(
            self.writer, application.application_id, configuration, change_summary="Initial release"
        )
        second = self.service.create_knowledge_revision(
            self.writer, application.application_id, configuration, change_summary="Policy update"
        )
        published = self.service.publish(self.writer, application.application_id, second.revision_id)
        rolled_back = self.service.rollback(self.writer, application.application_id, first.revision_id)

        self.assertEqual(first.revision_number, 1)
        self.assertEqual(second.revision_number, 2)
        self.assertEqual(published.application.active_revision_id, second.revision_id)
        self.assertEqual(rolled_back.application.active_revision_id, first.revision_id)
        self.assertEqual(
            tuple(item.revision_id for item in self.store.list_revisions(self.writer, application.application_id)),
            (second.revision_id, first.revision_id),
        )
        self.assertEqual(len(self.store.list_deployments(self.writer, application.application_id)), 2)
        self.assertEqual(len(self.store.list_audit_events(self.writer)), 6)

    def test_revision_and_publish_require_ready_bound_resources(self) -> None:
        project = self.service.create_project(self.writer, "Support")
        application = self.service.create_knowledge_application(
            self.writer, project.project_id, "Support assistant"
        )
        unavailable = ApplicationService(
            self.store, KnowledgeBaseStub(KnowledgeBaseStatus.INDEXING), clock=lambda: 10.0
        )
        configuration = KnowledgeChatConfiguration(knowledge_base_ids=(KNOWLEDGE_BASE_ID,))

        with self.assertRaises(ApplicationResourceUnavailableError):
            unavailable.create_knowledge_revision(
                self.writer, application.application_id, configuration, change_summary="Blocked"
            )

    def test_mutations_require_existing_writer_and_operator_roles(self) -> None:
        with self.assertRaises(ApplicationAuthorizationError):
            self.service.create_project(self.reader, "Forbidden")
        project = self.service.create_project(self.writer, "Support")
        application = self.service.create_knowledge_application(
            self.writer, project.project_id, "Support assistant"
        )
        revision = self.service.create_knowledge_revision(
            self.writer,
            application.application_id,
            KnowledgeChatConfiguration(knowledge_base_ids=(KNOWLEDGE_BASE_ID,)),
            change_summary="Initial release",
        )
        with self.assertRaises(ApplicationAuthorizationError):
            self.service.publish(self.reader, application.application_id, revision.revision_id)

    def test_archive_retains_history_and_records_an_audit_event(self) -> None:
        project = self.service.create_project(self.writer, "Support")
        application = self.service.create_knowledge_application(
            self.writer, project.project_id, "Support assistant"
        )
        revision = self.service.create_knowledge_revision(
            self.writer, application.application_id,
            KnowledgeChatConfiguration(knowledge_base_ids=(KNOWLEDGE_BASE_ID,)),
            change_summary="Initial release",
        )
        self.service.publish(self.writer, application.application_id, revision.revision_id)

        archived = self.service.archive_application(self.writer, application.application_id)

        self.assertEqual(archived.status, ApplicationStatus.ARCHIVED)
        self.assertEqual(archived.active_revision_id, revision.revision_id)
        self.assertEqual(
            self.store.list_audit_events(self.writer)[0].event_type,
            ApplicationAuditEventType.APPLICATION_ARCHIVED,
        )
        self.assertEqual(len(self.store.list_revisions(self.writer, application.application_id)), 1)
        self.assertEqual(len(self.store.list_deployments(self.writer, application.application_id)), 1)
        with self.assertRaises(ApplicationServiceValidationError):
            self.service.create_knowledge_revision(
                self.writer,
                application.application_id,
                KnowledgeChatConfiguration(knowledge_base_ids=(KNOWLEDGE_BASE_ID,)),
                change_summary="Must remain retired",
            )
        with self.assertRaises(ApplicationServiceValidationError):
            self.service.publish(
                self.writer, application.application_id, revision.revision_id
            )


if __name__ == "__main__":
    unittest.main()
