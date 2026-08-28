from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError

from rag_system.application_contracts import (
    APPLICATION_CONFIGURATION_SCHEMA_VERSION,
    AnswerPolicy,
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
    RetrievalProfile,
    SessionPolicy,
)
from rag_system.tenancy import TenantId


TENANT = TenantId("tenant-alpha")
PROJECT_ID = "prj_12345678901234567890123456789012"
APPLICATION_ID = "app_12345678901234567890123456789012"
REVISION_ID = "rev_12345678901234567890123456789012"
DEPLOYMENT_ID = "dep_12345678901234567890123456789012"
BINDING_ID = "bind_12345678901234567890123456789012"
AUDIT_EVENT_ID = "audit_12345678901234567890123456789012"
KNOWLEDGE_BASE_ID = "kb_12345678901234567890123456789012"


class ApplicationContractTests(unittest.TestCase):
    def test_typed_knowledge_chat_configuration_normalizes_and_is_immutable(self) -> None:
        configuration = KnowledgeChatConfiguration(
            knowledge_base_ids=[KNOWLEDGE_BASE_ID],
            answer_policy=AnswerPolicy(require_citations=True, allow_cloud=True),
            session_policy=SessionPolicy(ttl_seconds=3_600),
        )

        self.assertEqual(configuration.knowledge_base_ids, (KNOWLEDGE_BASE_ID,))
        self.assertTrue(configuration.answer_policy.allow_cloud)
        self.assertEqual(configuration.retrieval_profile, RetrievalProfile.DEFAULT)
        with self.assertRaises(FrozenInstanceError):
            configuration.knowledge_base_ids = ()  # type: ignore[misc]

    def test_knowledge_chat_configuration_rejects_untyped_or_duplicate_resources(self) -> None:
        with self.assertRaises(ApplicationValidationError):
            KnowledgeChatConfiguration(knowledge_base_ids="not-a-sequence")
        with self.assertRaises(ApplicationValidationError):
            KnowledgeChatConfiguration(knowledge_base_ids=(KNOWLEDGE_BASE_ID, KNOWLEDGE_BASE_ID))
        with self.assertRaises(ApplicationValidationError):
            KnowledgeChatConfiguration(knowledge_base_ids=("not-a-kb",))
        with self.assertRaises(ApplicationValidationError):
            KnowledgeChatConfiguration(knowledge_base_ids=(KNOWLEDGE_BASE_ID,), answer_policy={})  # type: ignore[arg-type]
        with self.assertRaises(ApplicationValidationError):
            KnowledgeChatConfiguration(
                knowledge_base_ids=(KNOWLEDGE_BASE_ID,), retrieval_profile="default"
            )  # type: ignore[arg-type]

    def test_policies_reject_ambiguous_values(self) -> None:
        with self.assertRaises(ApplicationValidationError):
            AnswerPolicy(require_citations=1)  # type: ignore[arg-type]
        with self.assertRaises(ApplicationValidationError):
            SessionPolicy(enabled=False, ttl_seconds=60)
        with self.assertRaises(ApplicationValidationError):
            SessionPolicy(ttl_seconds=59)

    def test_project_and_application_validate_tenant_lifecycle_and_timestamps(self) -> None:
        project = Project(
            project_id=PROJECT_ID,
            tenant_id=TENANT,
            display_name=" Support knowledge ",
            description="Private support corpus.",
            created_at=1.0,
            updated_at=2.0,
        )
        application = Application(
            application_id=APPLICATION_ID,
            tenant_id=TENANT,
            project_id=PROJECT_ID,
            display_name="Support assistant",
            application_kind=ApplicationKind.KNOWLEDGE_CHAT,
            active_revision_id=None,
            status=ApplicationStatus.ACTIVE,
            created_at=1.0,
            updated_at=2.0,
        )

        self.assertEqual(project.display_name, "Support knowledge")
        self.assertEqual(application.application_kind, ApplicationKind.KNOWLEDGE_CHAT)
        with self.assertRaises(ApplicationValidationError):
            Application(
                application_id=APPLICATION_ID,
                tenant_id=TENANT,
                project_id=PROJECT_ID,
                display_name="Assistant",
                application_kind="knowledge_chat",  # type: ignore[arg-type]
                active_revision_id=None,
                status=ApplicationStatus.ACTIVE,
                created_at=1.0,
                updated_at=0.0,
            )

    def test_revision_requires_current_schema_and_typed_configuration(self) -> None:
        configuration = KnowledgeChatConfiguration(knowledge_base_ids=(KNOWLEDGE_BASE_ID,))
        revision = ApplicationRevision(
            revision_id=REVISION_ID,
            application_id=APPLICATION_ID,
            revision_number=1,
            configuration_schema_version=APPLICATION_CONFIGURATION_SCHEMA_VERSION,
            configuration=configuration,
            created_at=3.0,
            created_by="operator@example.com",
            change_summary="Initial revision",
        )

        self.assertEqual(revision.revision_number, 1)
        with self.assertRaises(ApplicationValidationError):
            ApplicationRevision(
                revision_id=REVISION_ID,
                application_id=APPLICATION_ID,
                revision_number=1,
                configuration_schema_version=2,
                configuration=configuration,
                created_at=3.0,
                created_by="operator@example.com",
                change_summary="Initial revision",
            )
        with self.assertRaises(ApplicationValidationError):
            ApplicationRevision(
                revision_id=REVISION_ID,
                application_id=APPLICATION_ID,
                revision_number=1,
                configuration_schema_version=True,  # type: ignore[arg-type]
                configuration=configuration,
                created_at=3.0,
                created_by="operator@example.com",
                change_summary="Initial revision",
            )
        with self.assertRaises(ApplicationValidationError):
            ApplicationRevision(
                revision_id=REVISION_ID,
                application_id=APPLICATION_ID,
                revision_number=0,
                configuration_schema_version=APPLICATION_CONFIGURATION_SCHEMA_VERSION,
                configuration={},  # type: ignore[arg-type]
                created_at=3.0,
                created_by="operator@example.com",
                change_summary="Initial revision",
            )

    def test_deployment_and_binding_are_revision_scoped(self) -> None:
        deployment = Deployment(
            deployment_id=DEPLOYMENT_ID,
            application_id=APPLICATION_ID,
            revision_id=REVISION_ID,
            environment=DeploymentEnvironment.PRODUCTION,
            deployed_at=4.0,
            deployed_by="operator@example.com",
        )
        binding = ResourceBinding(
            binding_id=BINDING_ID,
            application_id=APPLICATION_ID,
            revision_id=REVISION_ID,
            resource_kind=ResourceKind.KNOWLEDGE_BASE,
            resource_id=KNOWLEDGE_BASE_ID,
            access_mode=ResourceAccessMode.READ,
            created_at=4.0,
        )

        self.assertEqual(deployment.environment, DeploymentEnvironment.PRODUCTION)
        self.assertEqual(binding.resource_id, KNOWLEDGE_BASE_ID)
        with self.assertRaises(ApplicationValidationError):
            ResourceBinding(
                binding_id=BINDING_ID,
                application_id=APPLICATION_ID,
                revision_id=REVISION_ID,
                resource_kind=ResourceKind.KNOWLEDGE_BASE,
                resource_id="not-a-kb",
                access_mode=ResourceAccessMode.READ,
                created_at=4.0,
            )

    def test_audit_events_are_tenant_bound_and_allow_only_safe_metadata(self) -> None:
        event = AuditEvent(
            audit_event_id=AUDIT_EVENT_ID,
            tenant_id=TENANT,
            event_type=ApplicationAuditEventType.REVISION_CREATED,
            occurred_at=5.0,
            actor="operator@example.com",
            summary="Created the first revision.",
            project_id=PROJECT_ID,
            application_id=APPLICATION_ID,
            revision_id=REVISION_ID,
        )

        self.assertEqual(event.event_type, ApplicationAuditEventType.REVISION_CREATED)
        with self.assertRaises(ApplicationValidationError):
            AuditEvent(
                audit_event_id=AUDIT_EVENT_ID,
                tenant_id=TENANT,
                event_type="revision_created",  # type: ignore[arg-type]
                occurred_at=5.0,
                actor="operator@example.com",
                summary="Unsafe\x01 detail",
            )

    def test_contract_rejects_non_finite_timestamps_and_unsafe_metadata(self) -> None:
        with self.assertRaises(ApplicationValidationError):
            Project(
                project_id=PROJECT_ID,
                tenant_id=TENANT,
                display_name="Unsafe/name",
                description="",
                created_at=0.0,
                updated_at=0.0,
            )
        with self.assertRaises(ApplicationValidationError):
            Deployment(
                deployment_id=DEPLOYMENT_ID,
                application_id=APPLICATION_ID,
                revision_id=REVISION_ID,
                environment=DeploymentEnvironment.PRODUCTION,
                deployed_at=math.inf,
                deployed_by="operator@example.com",
            )


if __name__ == "__main__":
    unittest.main()
