from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from rag_system.application_contracts import (
    APPLICATION_CONFIGURATION_SCHEMA_VERSION,
    Application,
    ApplicationAuditEventType,
    ApplicationKind,
    ApplicationRevision,
    ApplicationStatus,
    ApplicationValidationError,
    AuditEvent,
    Deployment,
    DeploymentEnvironment,
    KnowledgeChatConfiguration,
    Project,
    ResourceAccessMode,
    ResourceBinding,
    ResourceKind,
)
from rag_system.application_store import (
    ApplicationRevisionUnavailableError,
    ApplicationStore,
    ApplicationStoreSchemaError,
    ApplicationStoreStorageError,
    ApplicationUnavailableError,
    ProjectUnavailableError,
)
from rag_system.tenancy import Principal, TenantId


def make_principal(tenant: str) -> Principal:
    return Principal(f"operator-{tenant}", TenantId(tenant), frozenset({"operator"}))


def identifier(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix:0>32}"[-(len(prefix) + 33) :]


class ApplicationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name, "applications.sqlite3")
        self.store = ApplicationStore(self.database)
        self.tenant_a = make_principal("tenant-a")
        self.tenant_b = make_principal("tenant-b")

    def project(self, *, suffix: str = "1") -> Project:
        return Project(
            project_id=identifier("prj", suffix),
            tenant_id=self.tenant_a.tenant_id,
            display_name="Support automation",
            description="Support knowledge applications.",
            created_at=1.0,
            updated_at=1.0,
        )

    def application(self, project: Project, *, suffix: str = "1") -> Application:
        return Application(
            application_id=identifier("app", suffix),
            tenant_id=self.tenant_a.tenant_id,
            project_id=project.project_id,
            display_name="Support assistant",
            application_kind=ApplicationKind.KNOWLEDGE_CHAT,
            active_revision_id=None,
            status=ApplicationStatus.ACTIVE,
            created_at=2.0,
            updated_at=2.0,
        )

    def revision_and_binding(
        self, application: Application, *, suffix: str = "1", revision_number: int = 1
    ) -> tuple[ApplicationRevision, ResourceBinding]:
        knowledge_base_id = identifier("kb", suffix)
        revision = ApplicationRevision(
            revision_id=identifier("rev", suffix),
            application_id=application.application_id,
            revision_number=revision_number,
            configuration_schema_version=APPLICATION_CONFIGURATION_SCHEMA_VERSION,
            configuration=KnowledgeChatConfiguration(knowledge_base_ids=(knowledge_base_id,)),
            created_at=3.0,
            created_by=self.tenant_a.subject,
            change_summary="Initial support application.",
        )
        binding = ResourceBinding(
            binding_id=identifier("bind", suffix),
            application_id=application.application_id,
            revision_id=revision.revision_id,
            resource_kind=ResourceKind.KNOWLEDGE_BASE,
            resource_id=knowledge_base_id,
            access_mode=ResourceAccessMode.READ,
            created_at=3.0,
        )
        return revision, binding

    def test_persists_tenant_scoped_application_graph_across_restart(self) -> None:
        project = self.store.create_project(self.tenant_a, self.project())
        application = self.store.create_application(self.tenant_a, self.application(project))
        revision, binding = self.revision_and_binding(application)
        self.store.create_revision(self.tenant_a, revision, (binding,))
        deployment = Deployment(
            deployment_id=identifier("dep", "1"),
            application_id=application.application_id,
            revision_id=revision.revision_id,
            environment=DeploymentEnvironment.PRODUCTION,
            deployed_at=4.0,
            deployed_by=self.tenant_a.subject,
        )
        self.store.create_deployment(self.tenant_a, deployment)
        event = AuditEvent(
            audit_event_id=identifier("audit", "1"),
            tenant_id=self.tenant_a.tenant_id,
            event_type=ApplicationAuditEventType.DEPLOYMENT_CREATED,
            occurred_at=4.0,
            actor=self.tenant_a.subject,
            summary="Published the initial production revision.",
            project_id=project.project_id,
            application_id=application.application_id,
            revision_id=revision.revision_id,
        )
        self.store.record_audit_event(self.tenant_a, event)

        reopened = ApplicationStore(self.database)
        self.assertEqual(reopened.get_project(self.tenant_a, project.project_id), project)
        self.assertEqual(reopened.get_application(self.tenant_a, application.application_id), application)
        self.assertEqual(
            reopened.get_revision(self.tenant_a, application.application_id, revision.revision_id),
            revision,
        )
        self.assertEqual(
            reopened.list_bindings(self.tenant_a, application.application_id, revision.revision_id),
            (binding,),
        )
        self.assertEqual(reopened.list_deployments(self.tenant_a, application.application_id), (deployment,))
        self.assertEqual(reopened.list_audit_events(self.tenant_a), (event,))

    def test_tenant_and_invalid_resource_lookups_do_not_disclose_existence(self) -> None:
        project = self.store.create_project(self.tenant_a, self.project())
        application = self.store.create_application(self.tenant_a, self.application(project))
        revision, binding = self.revision_and_binding(application)
        self.store.create_revision(self.tenant_a, revision, (binding,))

        with self.assertRaises(ProjectUnavailableError):
            self.store.get_project(self.tenant_b, project.project_id)
        with self.assertRaises(ApplicationUnavailableError):
            self.store.get_application(self.tenant_b, application.application_id)
        with self.assertRaises(ApplicationUnavailableError):
            self.store.get_application(self.tenant_a, "not-an-application")
        with self.assertRaises(ApplicationRevisionUnavailableError):
            self.store.get_revision(self.tenant_b, application.application_id, revision.revision_id)

    def test_revision_bindings_must_exactly_match_typed_configuration(self) -> None:
        project = self.store.create_project(self.tenant_a, self.project())
        application = self.store.create_application(self.tenant_a, self.application(project))
        revision, binding = self.revision_and_binding(application)
        unmatched = ResourceBinding(
            binding_id=identifier("bind", "2"),
            application_id=application.application_id,
            revision_id=revision.revision_id,
            resource_kind=ResourceKind.KNOWLEDGE_BASE,
            resource_id=identifier("kb", "2"),
            access_mode=ResourceAccessMode.READ,
            created_at=3.0,
        )

        with self.assertRaisesRegex(ApplicationValidationError, "exactly match"):
            self.store.create_revision(self.tenant_a, revision, (unmatched,))
        self.assertEqual(self.store.list_revisions(self.tenant_a, application.application_id), ())

    def test_schema_rejects_unknown_version_and_corrupted_configuration(self) -> None:
        project = self.store.create_project(self.tenant_a, self.project())
        application = self.store.create_application(self.tenant_a, self.application(project))
        revision, binding = self.revision_and_binding(application)
        self.store.create_revision(self.tenant_a, revision, (binding,))
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("UPDATE application_revisions SET configuration_json = '{}' ")
            connection.commit()

        with self.assertRaises(ApplicationStoreSchemaError):
            self.store.get_revision(self.tenant_a, application.application_id, revision.revision_id)

        replacement = Path(self.directory.name, "unknown-version.sqlite3")
        with closing(sqlite3.connect(replacement)) as connection:
            connection.execute("PRAGMA user_version = 99")
            connection.commit()
        with self.assertRaises(ApplicationStoreSchemaError):
            ApplicationStore(replacement)

    def test_schema_uses_wal_and_rejects_duplicate_records(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")
            self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 0)
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
        project = self.project()
        self.store.create_project(self.tenant_a, project)
        with self.assertRaises(ApplicationStoreStorageError):
            self.store.create_project(self.tenant_a, project)

    def test_schema_rejects_a_missing_custom_index_without_rewriting_data(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("DROP INDEX idx_revisions_application_number")
            connection.commit()

        with self.assertRaises(ApplicationStoreSchemaError):
            ApplicationStore(self.database)
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertIsNone(
                connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index' "
                    "AND name = 'idx_revisions_application_number'"
                ).fetchone()
            )

    def test_publish_is_atomic_when_deployment_or_audit_persistence_fails(self) -> None:
        project = self.store.create_project(self.tenant_a, self.project())
        application = self.store.create_application(self.tenant_a, self.application(project))
        first_revision, first_binding = self.revision_and_binding(application, suffix="1")
        second_revision, second_binding = self.revision_and_binding(
            application, suffix="2", revision_number=2
        )
        self.store.create_revision(self.tenant_a, first_revision, (first_binding,))
        self.store.create_revision(self.tenant_a, second_revision, (second_binding,))
        first_deployment = Deployment(
            deployment_id=identifier("dep", "1"),
            application_id=application.application_id,
            revision_id=first_revision.revision_id,
            environment=DeploymentEnvironment.PRODUCTION,
            deployed_at=4.0,
            deployed_by=self.tenant_a.subject,
        )
        first_event = AuditEvent(
            audit_event_id=identifier("audit", "1"),
            tenant_id=self.tenant_a.tenant_id,
            event_type=ApplicationAuditEventType.DEPLOYMENT_CREATED,
            occurred_at=4.0,
            actor=self.tenant_a.subject,
            summary="Published the first revision.",
            project_id=project.project_id,
            application_id=application.application_id,
            revision_id=first_revision.revision_id,
        )
        published = self.store.publish(
            self.tenant_a, first_deployment, first_event, updated_at=4.0
        )
        duplicate_deployment = Deployment(
            deployment_id=first_deployment.deployment_id,
            application_id=application.application_id,
            revision_id=second_revision.revision_id,
            environment=DeploymentEnvironment.PRODUCTION,
            deployed_at=5.0,
            deployed_by=self.tenant_a.subject,
        )
        second_event = AuditEvent(
            audit_event_id=identifier("audit", "2"),
            tenant_id=self.tenant_a.tenant_id,
            event_type=ApplicationAuditEventType.DEPLOYMENT_CREATED,
            occurred_at=5.0,
            actor=self.tenant_a.subject,
            summary="Published the second revision.",
            project_id=project.project_id,
            application_id=application.application_id,
            revision_id=second_revision.revision_id,
        )

        self.assertEqual(published.active_revision_id, first_revision.revision_id)
        with self.assertRaises(ApplicationStoreStorageError):
            self.store.publish(self.tenant_a, duplicate_deployment, second_event, updated_at=5.0)
        active = self.store.get_application(self.tenant_a, application.application_id)
        self.assertEqual(active.active_revision_id, first_revision.revision_id)
        self.assertEqual(self.store.list_deployments(self.tenant_a, application.application_id), (first_deployment,))
        self.assertEqual(self.store.list_audit_events(self.tenant_a), (first_event,))


if __name__ == "__main__":
    unittest.main()
