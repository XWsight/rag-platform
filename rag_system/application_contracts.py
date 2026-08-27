"""Storage-neutral contracts for versioned applications.

The application kernel deliberately models configuration as typed values instead
of an unbounded JSON blob.  This keeps application revisions inspectable,
validatable, and safe to persist before a runtime or HTTP surface is added.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from rag_system.knowledge_base_contracts import validate_resource_id
from rag_system.tenancy import TenantId


APPLICATION_CONFIGURATION_SCHEMA_VERSION = 1
MAX_KNOWLEDGE_BASE_BINDINGS = 32

_PROJECT_ID_PATTERN = re.compile(r"prj_[A-Za-z0-9_-]{32}")
_APPLICATION_ID_PATTERN = re.compile(r"app_[A-Za-z0-9_-]{32}")
_REVISION_ID_PATTERN = re.compile(r"rev_[A-Za-z0-9_-]{32}")
_DEPLOYMENT_ID_PATTERN = re.compile(r"dep_[A-Za-z0-9_-]{32}")
_BINDING_ID_PATTERN = re.compile(r"bind_[A-Za-z0-9_-]{32}")
_SUBJECT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/-]{0,254}")


class ApplicationContractError(Exception):
    """Base error for invalid application-kernel contract values."""


class ApplicationValidationError(ApplicationContractError, ValueError):
    """An application-kernel value violates its storage-neutral contract."""


class ApplicationKind(StrEnum):
    """Application runtimes supported by the first platform kernel."""

    KNOWLEDGE_CHAT = "knowledge_chat"


class ApplicationStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class DeploymentEnvironment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ResourceKind(StrEnum):
    KNOWLEDGE_BASE = "knowledge_base"


class ResourceAccessMode(StrEnum):
    READ = "read"


@dataclass(frozen=True, slots=True)
class AnswerPolicy:
    """Answer behaviour that can be audited as part of a revision."""

    require_citations: bool = True
    allow_web: bool = False
    allow_research: bool = False

    def __post_init__(self) -> None:
        for field_name in ("require_citations", "allow_web", "allow_research"):
            if not isinstance(getattr(self, field_name), bool):
                raise ApplicationValidationError(f"{field_name} must be a boolean.")


@dataclass(frozen=True, slots=True)
class SessionPolicy:
    """Conversation retention policy for an application revision."""

    enabled: bool = True
    ttl_seconds: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ApplicationValidationError("enabled must be a boolean.")
        if self.ttl_seconds is not None:
            if isinstance(self.ttl_seconds, bool) or not isinstance(self.ttl_seconds, int):
                raise ApplicationValidationError("ttl_seconds must be an integer or None.")
            if not 60 <= self.ttl_seconds <= 2_592_000:
                raise ApplicationValidationError(
                    "ttl_seconds must be between one minute and thirty days."
                )
        if not self.enabled and self.ttl_seconds is not None:
            raise ApplicationValidationError("Disabled sessions cannot define a retention period.")


@dataclass(frozen=True, slots=True)
class KnowledgeChatConfiguration:
    """Typed configuration for the initial trusted knowledge-chat application."""

    knowledge_base_ids: tuple[str, ...]
    answer_policy: AnswerPolicy = AnswerPolicy()
    session_policy: SessionPolicy = SessionPolicy()

    def __post_init__(self) -> None:
        if isinstance(self.knowledge_base_ids, (str, bytes)) or not isinstance(
            self.knowledge_base_ids, Sequence
        ):
            raise ApplicationValidationError("knowledge_base_ids must be a sequence of IDs.")
        normalized = tuple(self.knowledge_base_ids)
        if not 1 <= len(normalized) <= MAX_KNOWLEDGE_BASE_BINDINGS:
            raise ApplicationValidationError("knowledge_base_ids has an invalid item count.")
        try:
            normalized = tuple(validate_resource_id(item) for item in normalized)
        except ValueError as error:
            raise ApplicationValidationError("knowledge_base_ids contains an invalid ID.") from error
        if len(set(normalized)) != len(normalized):
            raise ApplicationValidationError("knowledge_base_ids cannot contain duplicates.")
        if not isinstance(self.answer_policy, AnswerPolicy):
            raise ApplicationValidationError("answer_policy must be an AnswerPolicy.")
        if not isinstance(self.session_policy, SessionPolicy):
            raise ApplicationValidationError("session_policy must be a SessionPolicy.")
        object.__setattr__(self, "knowledge_base_ids", normalized)


@dataclass(frozen=True, slots=True)
class Project:
    project_id: str
    tenant_id: TenantId
    display_name: str
    description: str
    created_at: float
    updated_at: float

    def __post_init__(self) -> None:
        validate_project_id(self.project_id)
        _validate_tenant_id(self.tenant_id)
        object.__setattr__(self, "display_name", validate_display_name(self.display_name))
        object.__setattr__(self, "description", validate_description(self.description))
        validate_time_range(self.created_at, self.updated_at)


@dataclass(frozen=True, slots=True)
class Application:
    application_id: str
    tenant_id: TenantId
    project_id: str
    display_name: str
    application_kind: ApplicationKind
    active_revision_id: str | None
    status: ApplicationStatus
    created_at: float
    updated_at: float

    def __post_init__(self) -> None:
        validate_application_id(self.application_id)
        _validate_tenant_id(self.tenant_id)
        validate_project_id(self.project_id)
        object.__setattr__(self, "display_name", validate_display_name(self.display_name))
        if not isinstance(self.application_kind, ApplicationKind):
            raise ApplicationValidationError("application_kind must be an ApplicationKind.")
        if self.active_revision_id is not None:
            validate_revision_id(self.active_revision_id)
        if not isinstance(self.status, ApplicationStatus):
            raise ApplicationValidationError("status must be an ApplicationStatus.")
        validate_time_range(self.created_at, self.updated_at)


@dataclass(frozen=True, slots=True)
class ApplicationRevision:
    revision_id: str
    application_id: str
    revision_number: int
    configuration_schema_version: int
    configuration: KnowledgeChatConfiguration
    created_at: float
    created_by: str
    change_summary: str

    def __post_init__(self) -> None:
        validate_revision_id(self.revision_id)
        validate_application_id(self.application_id)
        if isinstance(self.revision_number, bool) or not isinstance(self.revision_number, int):
            raise ApplicationValidationError("revision_number must be a positive integer.")
        if self.revision_number < 1:
            raise ApplicationValidationError("revision_number must be a positive integer.")
        if (
            isinstance(self.configuration_schema_version, bool)
            or not isinstance(self.configuration_schema_version, int)
            or self.configuration_schema_version != APPLICATION_CONFIGURATION_SCHEMA_VERSION
        ):
            raise ApplicationValidationError("Unsupported application configuration schema version.")
        if not isinstance(self.configuration, KnowledgeChatConfiguration):
            raise ApplicationValidationError(
                "knowledge_chat revisions require a KnowledgeChatConfiguration."
            )
        if not is_valid_timestamp(self.created_at):
            raise ApplicationValidationError("created_at must be finite and non-negative.")
        object.__setattr__(self, "created_by", validate_subject(self.created_by))
        object.__setattr__(self, "change_summary", validate_change_summary(self.change_summary))


@dataclass(frozen=True, slots=True)
class Deployment:
    deployment_id: str
    application_id: str
    revision_id: str
    environment: DeploymentEnvironment
    deployed_at: float
    deployed_by: str

    def __post_init__(self) -> None:
        validate_deployment_id(self.deployment_id)
        validate_application_id(self.application_id)
        validate_revision_id(self.revision_id)
        if not isinstance(self.environment, DeploymentEnvironment):
            raise ApplicationValidationError("environment must be a DeploymentEnvironment.")
        if not is_valid_timestamp(self.deployed_at):
            raise ApplicationValidationError("deployed_at must be finite and non-negative.")
        object.__setattr__(self, "deployed_by", validate_subject(self.deployed_by))


@dataclass(frozen=True, slots=True)
class ResourceBinding:
    binding_id: str
    application_id: str
    revision_id: str
    resource_kind: ResourceKind
    resource_id: str
    access_mode: ResourceAccessMode
    created_at: float

    def __post_init__(self) -> None:
        validate_binding_id(self.binding_id)
        validate_application_id(self.application_id)
        validate_revision_id(self.revision_id)
        if not isinstance(self.resource_kind, ResourceKind):
            raise ApplicationValidationError("resource_kind must be a ResourceKind.")
        if self.resource_kind is ResourceKind.KNOWLEDGE_BASE:
            try:
                object.__setattr__(self, "resource_id", validate_resource_id(self.resource_id))
            except ValueError as error:
                raise ApplicationValidationError("resource_id must be a knowledge base ID.") from error
        if not isinstance(self.access_mode, ResourceAccessMode):
            raise ApplicationValidationError("access_mode must be a ResourceAccessMode.")
        if (
            self.resource_kind is ResourceKind.KNOWLEDGE_BASE
            and self.access_mode is not ResourceAccessMode.READ
        ):
            raise ApplicationValidationError("Knowledge bases may only be bound with read access.")
        if not is_valid_timestamp(self.created_at):
            raise ApplicationValidationError("created_at must be finite and non-negative.")


def validate_project_id(value: object) -> str:
    return _validate_identifier(value, _PROJECT_ID_PATTERN, "project ID")


def validate_application_id(value: object) -> str:
    return _validate_identifier(value, _APPLICATION_ID_PATTERN, "application ID")


def validate_revision_id(value: object) -> str:
    return _validate_identifier(value, _REVISION_ID_PATTERN, "revision ID")


def validate_deployment_id(value: object) -> str:
    return _validate_identifier(value, _DEPLOYMENT_ID_PATTERN, "deployment ID")


def validate_binding_id(value: object) -> str:
    return _validate_identifier(value, _BINDING_ID_PATTERN, "binding ID")


def validate_display_name(value: object) -> str:
    if not isinstance(value, str):
        raise ApplicationValidationError("Display name must be text.")
    normalized = value.strip()
    if not normalized or len(normalized) > 200:
        raise ApplicationValidationError("Display name has an invalid length.")
    if any(ord(character) < 32 for character in normalized) or any(
        character in '/\\<>:"|?*' for character in normalized
    ):
        raise ApplicationValidationError("Display name contains unsafe characters.")
    if normalized.endswith((".", " ")):
        raise ApplicationValidationError("Display name has an unsafe ending.")
    return normalized


def validate_description(value: object) -> str:
    if not isinstance(value, str) or len(value) > 2_000:
        raise ApplicationValidationError("Description has an invalid length.")
    if any(ord(character) < 32 and character not in {"\n", "\r", "\t"} for character in value):
        raise ApplicationValidationError("Description contains unsupported control characters.")
    return value


def validate_change_summary(value: object) -> str:
    if not isinstance(value, str):
        raise ApplicationValidationError("Change summary must be text.")
    normalized = value.strip()
    if not normalized or len(normalized) > 1_000:
        raise ApplicationValidationError("Change summary has an invalid length.")
    if any(ord(character) < 32 and character not in {"\n", "\r", "\t"} for character in normalized):
        raise ApplicationValidationError("Change summary contains unsupported control characters.")
    return normalized


def validate_subject(value: object) -> str:
    if not isinstance(value, str) or _SUBJECT_PATTERN.fullmatch(value) is None:
        raise ApplicationValidationError("Subject has an invalid format.")
    return value


def validate_time_range(created_at: object, updated_at: object) -> None:
    if not is_valid_timestamp(created_at) or not is_valid_timestamp(updated_at):
        raise ApplicationValidationError("Timestamps must be finite and non-negative.")
    if float(cast(int | float, updated_at)) < float(cast(int | float, created_at)):
        raise ApplicationValidationError("updated_at cannot precede created_at.")


def is_valid_timestamp(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _validate_identifier(value: object, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ApplicationValidationError(f"Invalid {label}.")
    return value


def _validate_tenant_id(value: object) -> TenantId:
    if not isinstance(value, TenantId):
        raise ApplicationValidationError("tenant_id must be a TenantId.")
    return value


__all__ = [
    "APPLICATION_CONFIGURATION_SCHEMA_VERSION",
    "MAX_KNOWLEDGE_BASE_BINDINGS",
    "AnswerPolicy",
    "Application",
    "ApplicationContractError",
    "ApplicationKind",
    "ApplicationRevision",
    "ApplicationStatus",
    "ApplicationValidationError",
    "Deployment",
    "DeploymentEnvironment",
    "KnowledgeChatConfiguration",
    "Project",
    "ResourceAccessMode",
    "ResourceBinding",
    "ResourceKind",
    "SessionPolicy",
    "is_valid_timestamp",
    "validate_application_id",
    "validate_binding_id",
    "validate_change_summary",
    "validate_deployment_id",
    "validate_description",
    "validate_display_name",
    "validate_project_id",
    "validate_revision_id",
    "validate_subject",
    "validate_time_range",
]
