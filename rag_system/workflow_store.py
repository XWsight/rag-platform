"""Durable, tenant-scoped SQLite storage for versioned workflow resources."""

from __future__ import annotations

import json
import sqlite3
from contextlib import AbstractContextManager
from pathlib import Path
from threading import RLock
from typing import Any

from rag_system.sqlite_support import SqliteDatabase
from rag_system.tenancy import Principal, TenantId
from rag_system.workflow_models import (
    ApprovalDecision,
    ExecutionBudget,
    Workflow,
    WorkflowApproval,
    WorkflowDeployment,
    WorkflowDeploymentStatus,
    WorkflowDraft,
    WorkflowModelError,
    WorkflowRevision,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowStatus,
    WorkflowStepRun,
    WorkflowStepStatus,
    validate_workflow_id,
    validate_workflow_revision_id,
    validate_workflow_run_id,
)
from rag_system.workflow_contracts import WorkflowSpec, WorkflowValidationError


_SCHEMA_VERSION = 1
_MAX_LIST_LIMIT = 100


class WorkflowStoreError(WorkflowModelError):
    """Base class for workflow-store failures."""


class WorkflowStoreSchemaError(WorkflowStoreError):
    def __init__(self) -> None:
        super().__init__("Workflow store schema or stored data is invalid.")


class WorkflowStoreStorageError(WorkflowStoreError):
    def __init__(self) -> None:
        super().__init__("Workflow store operation failed.")


class WorkflowUnavailableError(WorkflowStoreError):
    def __init__(self) -> None:
        super().__init__("Workflow is unavailable.")


class WorkflowRevisionUnavailableError(WorkflowStoreError):
    def __init__(self) -> None:
        super().__init__("Workflow revision is unavailable.")


class WorkflowRunUnavailableError(WorkflowStoreError):
    def __init__(self) -> None:
        super().__init__("Workflow run is unavailable.")


class WorkflowDraftConflictError(WorkflowStoreError):
    def __init__(self) -> None:
        super().__init__("Workflow draft has changed.")


class WorkflowPublishConflictError(WorkflowStoreError):
    def __init__(self) -> None:
        super().__init__("Workflow publication has changed.")


class WorkflowApprovalUnavailableError(WorkflowStoreError):
    def __init__(self) -> None:
        super().__init__("Workflow approval is unavailable.")


class WorkflowStore:
    """Use short transactions and tenant filters for all workflow state."""

    def __init__(self, database_path: str | Path, *, timeout_seconds: float = 5.0) -> None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise WorkflowModelError("workflow store timeout is invalid")
        if not 0 < timeout_seconds <= 60:
            raise WorkflowModelError("workflow store timeout is invalid")
        path = Path(database_path)
        if path.exists() and not path.is_file():
            raise WorkflowModelError("workflow database path must reference a file")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._database_path = path.resolve()
        self._database = SqliteDatabase(self._database_path, timeout_seconds=float(timeout_seconds))
        self._write_lock = RLock()
        self._initialize()

    @property
    def database_path(self) -> Path:
        return self._database_path

    def create_workflow(self, principal: Principal, workflow: Workflow) -> Workflow:
        tenant = _tenant(principal)
        _same_tenant(tenant, workflow.tenant_id)
        with self._write_lock, self._write() as connection:
            try:
                connection.execute(
                    """INSERT INTO workflows (
                    workflow_id, tenant_id, project_id, display_name, active_revision_id,
                    status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        workflow.workflow_id,
                        tenant.value,
                        workflow.project_id,
                        workflow.display_name,
                        workflow.active_revision_id,
                        workflow.status.value,
                        workflow.created_at,
                        workflow.updated_at,
                    ),
                )
                connection.execute(
                    """INSERT INTO workflow_drafts (
                    workflow_id, version, specification_json, budget_json, updated_at, updated_by, change_summary
                    ) VALUES (?, 0, NULL, NULL, ?, ?, '')""",
                    (workflow.workflow_id, workflow.updated_at, principal.subject),
                )
            except sqlite3.IntegrityError as error:
                raise WorkflowStoreStorageError() from error
        return workflow

    def get_workflow(self, principal: Principal, workflow_id: str) -> Workflow:
        tenant = _tenant(principal)
        clean_id = _safe_workflow_id(workflow_id)
        with self._read() as connection:
            row = connection.execute(
                "SELECT * FROM workflows WHERE workflow_id = ? AND tenant_id = ?", (clean_id, tenant.value)
            ).fetchone()
        if row is None:
            raise WorkflowUnavailableError()
        return _workflow_from_row(row)

    def list_workflows(
        self, principal: Principal, project_id: str, *, limit: int = 50
    ) -> tuple[Workflow, ...]:
        tenant = _tenant(principal)
        clean_limit = _limit(limit)
        with self._read() as connection:
            rows = connection.execute(
                """SELECT * FROM workflows WHERE tenant_id = ? AND project_id = ?
                ORDER BY updated_at DESC, workflow_id DESC LIMIT ?""",
                (tenant.value, project_id, clean_limit),
            ).fetchall()
        return tuple(_workflow_from_row(row) for row in rows)

    def archive_workflow(self, principal: Principal, workflow_id: str, *, updated_at: float) -> Workflow:
        tenant = _tenant(principal)
        clean_id = _safe_workflow_id(workflow_id)
        with self._write_lock, self._write() as connection:
            row = _require_workflow(connection, tenant, clean_id)
            workflow = _workflow_from_row(row)
            if workflow.status is WorkflowStatus.ARCHIVED:
                return workflow
            connection.execute(
                "UPDATE workflows SET status = ?, updated_at = ? WHERE workflow_id = ? AND tenant_id = ?",
                (WorkflowStatus.ARCHIVED.value, updated_at, clean_id, tenant.value),
            )
        return Workflow(
            workflow_id=workflow.workflow_id,
            tenant_id=workflow.tenant_id,
            project_id=workflow.project_id,
            display_name=workflow.display_name,
            active_revision_id=workflow.active_revision_id,
            status=WorkflowStatus.ARCHIVED,
            created_at=workflow.created_at,
            updated_at=updated_at,
        )

    def get_draft(self, principal: Principal, workflow_id: str) -> WorkflowDraft:
        tenant = _tenant(principal)
        clean_id = _safe_workflow_id(workflow_id)
        with self._read() as connection:
            _require_workflow(connection, tenant, clean_id)
            row = connection.execute(
                "SELECT * FROM workflow_drafts WHERE workflow_id = ?", (clean_id,)
            ).fetchone()
        if row is None:
            raise WorkflowStoreSchemaError()
        return _draft_from_row(row)

    def update_draft(
        self,
        principal: Principal,
        draft: WorkflowDraft,
        *,
        expected_version: int,
    ) -> WorkflowDraft:
        tenant = _tenant(principal)
        if not isinstance(expected_version, int) or isinstance(expected_version, bool) or expected_version < 0:
            raise WorkflowModelError("workflow draft expected version is invalid")
        with self._write_lock, self._write() as connection:
            _require_workflow(connection, tenant, draft.workflow_id)
            current = connection.execute(
                "SELECT version FROM workflow_drafts WHERE workflow_id = ?", (draft.workflow_id,)
            ).fetchone()
            if current is None:
                raise WorkflowStoreSchemaError()
            if int(current["version"]) != expected_version or draft.version != expected_version + 1:
                raise WorkflowDraftConflictError()
            connection.execute(
                """UPDATE workflow_drafts SET version = ?, specification_json = ?, budget_json = ?,
                updated_at = ?, updated_by = ?, change_summary = ? WHERE workflow_id = ?""",
                (
                    draft.version,
                    _specification_json(draft.specification),
                    _budget_json(draft.budget),
                    draft.updated_at,
                    draft.updated_by,
                    draft.change_summary,
                    draft.workflow_id,
                ),
            )
        return draft

    def create_revision(self, principal: Principal, revision: WorkflowRevision) -> WorkflowRevision:
        tenant = _tenant(principal)
        with self._write_lock, self._write() as connection:
            workflow = _workflow_from_row(_require_workflow(connection, tenant, revision.workflow_id))
            if workflow.status is WorkflowStatus.ARCHIVED:
                raise WorkflowUnavailableError()
            row = connection.execute(
                "SELECT COALESCE(MAX(revision_number), 0) AS current_number FROM workflow_revisions WHERE workflow_id = ?",
                (revision.workflow_id,),
            ).fetchone()
            if row is None or revision.revision_number != int(row["current_number"]) + 1:
                raise WorkflowStoreStorageError()
            try:
                connection.execute(
                    """INSERT INTO workflow_revisions (
                    revision_id, workflow_id, revision_number, specification_json, specification_digest,
                    budget_json, created_at, created_by, change_summary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        revision.revision_id,
                        revision.workflow_id,
                        revision.revision_number,
                        revision.specification.to_json(),
                        revision.specification_digest,
                        _budget_json(revision.budget),
                        revision.created_at,
                        revision.created_by,
                        revision.change_summary,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise WorkflowStoreStorageError() from error
        return revision

    def get_revision(
        self, principal: Principal, workflow_id: str, revision_id: str
    ) -> WorkflowRevision:
        tenant = _tenant(principal)
        clean_workflow_id = _safe_workflow_id(workflow_id)
        clean_revision_id = _safe_revision_id(revision_id)
        with self._read() as connection:
            _require_workflow(connection, tenant, clean_workflow_id)
            row = connection.execute(
                "SELECT * FROM workflow_revisions WHERE workflow_id = ? AND revision_id = ?",
                (clean_workflow_id, clean_revision_id),
            ).fetchone()
        if row is None:
            raise WorkflowRevisionUnavailableError()
        return _revision_from_row(row)

    def list_revisions(
        self, principal: Principal, workflow_id: str, *, limit: int = 50
    ) -> tuple[WorkflowRevision, ...]:
        tenant = _tenant(principal)
        clean_id = _safe_workflow_id(workflow_id)
        with self._read() as connection:
            _require_workflow(connection, tenant, clean_id)
            rows = connection.execute(
                """SELECT * FROM workflow_revisions WHERE workflow_id = ?
                ORDER BY revision_number DESC LIMIT ?""",
                (clean_id, _limit(limit)),
            ).fetchall()
        return tuple(_revision_from_row(row) for row in rows)

    def publish(
        self,
        principal: Principal,
        deployment: WorkflowDeployment,
        *,
        updated_at: float,
        expected_active_revision_id: str | None,
    ) -> Workflow:
        tenant = _tenant(principal)
        with self._write_lock, self._write() as connection:
            row = _require_workflow(connection, tenant, deployment.workflow_id)
            workflow = _workflow_from_row(row)
            if workflow.status is WorkflowStatus.ARCHIVED:
                raise WorkflowUnavailableError()
            if workflow.active_revision_id != expected_active_revision_id:
                raise WorkflowPublishConflictError()
            _require_revision(connection, deployment.workflow_id, deployment.revision_id)
            connection.execute(
                """UPDATE workflow_deployments SET status = ?
                WHERE workflow_id = ? AND status = ?""",
                (
                    WorkflowDeploymentStatus.SUPERSEDED.value,
                    deployment.workflow_id,
                    WorkflowDeploymentStatus.ACTIVE.value,
                ),
            )
            try:
                connection.execute(
                    """INSERT INTO workflow_deployments (
                    deployment_id, workflow_id, revision_id, deployed_at, deployed_by, status
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        deployment.deployment_id,
                        deployment.workflow_id,
                        deployment.revision_id,
                        deployment.deployed_at,
                        deployment.deployed_by,
                        WorkflowDeploymentStatus.ACTIVE.value,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise WorkflowStoreStorageError() from error
            connection.execute(
                """UPDATE workflows SET active_revision_id = ?, updated_at = ?
                WHERE workflow_id = ? AND tenant_id = ?""",
                (deployment.revision_id, updated_at, deployment.workflow_id, tenant.value),
            )
        return Workflow(
            workflow_id=workflow.workflow_id,
            tenant_id=workflow.tenant_id,
            project_id=workflow.project_id,
            display_name=workflow.display_name,
            active_revision_id=deployment.revision_id,
            status=workflow.status,
            created_at=workflow.created_at,
            updated_at=updated_at,
        )

    def create_run(self, principal: Principal, run: WorkflowRun) -> WorkflowRun:
        tenant = _tenant(principal)
        with self._write_lock, self._write() as connection:
            _require_workflow(connection, tenant, run.workflow_id)
            revision = _revision_from_row(_require_revision(connection, run.workflow_id, run.revision_id))
            if revision.specification_digest != run.specification_digest:
                raise WorkflowStoreStorageError()
            try:
                connection.execute(
                    """INSERT INTO workflow_runs (
                    run_id, workflow_id, revision_id, specification_digest, status, created_at,
                    updated_at, created_by, input_digest, error_code
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run.run_id,
                        run.workflow_id,
                        run.revision_id,
                        run.specification_digest,
                        run.status.value,
                        run.created_at,
                        run.updated_at,
                        run.created_by,
                        run.input_digest,
                        run.error_code,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise WorkflowStoreStorageError() from error
        return run

    def get_run(self, principal: Principal, run_id: str) -> WorkflowRun:
        tenant = _tenant(principal)
        clean_id = _safe_run_id(run_id)
        with self._read() as connection:
            row = connection.execute(
                """SELECT runs.* FROM workflow_runs AS runs
                JOIN workflows AS workflows ON workflows.workflow_id = runs.workflow_id
                WHERE runs.run_id = ? AND workflows.tenant_id = ?""",
                (clean_id, tenant.value),
            ).fetchone()
        if row is None:
            raise WorkflowRunUnavailableError()
        return _run_from_row(row)

    def transition_run(
        self,
        principal: Principal,
        run_id: str,
        *,
        status: WorkflowRunStatus,
        updated_at: float,
        error_code: str | None = None,
    ) -> WorkflowRun:
        if not isinstance(status, WorkflowRunStatus):
            raise WorkflowModelError("workflow run status is invalid")
        current = self.get_run(principal, run_id)
        _validate_run_transition(current.status, status)
        tenant = _tenant(principal)
        with self._write_lock, self._write() as connection:
            changed = connection.execute(
                """UPDATE workflow_runs SET status = ?, updated_at = ?, error_code = ?
                WHERE run_id = ? AND workflow_id IN (
                  SELECT workflow_id FROM workflows WHERE tenant_id = ?
                )""",
                (status.value, updated_at, error_code, current.run_id, tenant.value),
            ).rowcount
            if changed != 1:
                raise WorkflowRunUnavailableError()
        return WorkflowRun(
            run_id=current.run_id,
            workflow_id=current.workflow_id,
            revision_id=current.revision_id,
            specification_digest=current.specification_digest,
            status=status,
            created_at=current.created_at,
            updated_at=updated_at,
            created_by=current.created_by,
            input_digest=current.input_digest,
            error_code=error_code,
        )

    def save_step_run(self, principal: Principal, step: WorkflowStepRun) -> WorkflowStepRun:
        self.get_run(principal, step.run_id)
        with self._write_lock, self._write() as connection:
            try:
                connection.execute(
                    """INSERT INTO workflow_step_runs (
                    step_run_id, run_id, node_id, status, started_at, finished_at,
                    input_digest, output_digest, error_code
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(step_run_id) DO UPDATE SET
                    status = excluded.status, started_at = excluded.started_at,
                    finished_at = excluded.finished_at, input_digest = excluded.input_digest,
                    output_digest = excluded.output_digest, error_code = excluded.error_code""",
                    (
                        step.step_run_id,
                        step.run_id,
                        step.node_id,
                        step.status.value,
                        step.started_at,
                        step.finished_at,
                        step.input_digest,
                        step.output_digest,
                        step.error_code,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise WorkflowStoreStorageError() from error
        return step

    def list_step_runs(self, principal: Principal, run_id: str) -> tuple[WorkflowStepRun, ...]:
        self.get_run(principal, run_id)
        with self._read() as connection:
            rows = connection.execute(
                "SELECT * FROM workflow_step_runs WHERE run_id = ? ORDER BY step_run_id", (run_id,)
            ).fetchall()
        return tuple(_step_from_row(row) for row in rows)

    def create_approval(self, principal: Principal, approval: WorkflowApproval) -> WorkflowApproval:
        self.get_run(principal, approval.run_id)
        with self._write_lock, self._write() as connection:
            try:
                connection.execute(
                    """INSERT INTO workflow_approvals (
                    approval_id, run_id, node_id, requested_at, requested_by, decision, decided_at, decided_by
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        approval.approval_id,
                        approval.run_id,
                        approval.node_id,
                        approval.requested_at,
                        approval.requested_by,
                        None,
                        None,
                        None,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise WorkflowStoreStorageError() from error
        return approval

    def decide_approval(
        self,
        principal: Principal,
        approval_id: str,
        *,
        decision: ApprovalDecision,
        decided_at: float,
    ) -> WorkflowApproval:
        if not isinstance(decision, ApprovalDecision):
            raise WorkflowModelError("workflow approval decision is invalid")
        tenant = _tenant(principal)
        with self._write_lock, self._write() as connection:
            row = connection.execute(
                """SELECT approvals.* FROM workflow_approvals AS approvals
                JOIN workflow_runs AS runs ON runs.run_id = approvals.run_id
                JOIN workflows AS workflows ON workflows.workflow_id = runs.workflow_id
                WHERE approvals.approval_id = ? AND workflows.tenant_id = ?""",
                (approval_id, tenant.value),
            ).fetchone()
            if row is None or row["decision"] is not None:
                raise WorkflowApprovalUnavailableError()
            changed = connection.execute(
                """UPDATE workflow_approvals SET decision = ?, decided_at = ?, decided_by = ?
                WHERE approval_id = ? AND decision IS NULL""",
                (decision.value, decided_at, principal.subject, approval_id),
            ).rowcount
            if changed != 1:
                raise WorkflowApprovalUnavailableError()
        return WorkflowApproval(
            approval_id=str(row["approval_id"]),
            run_id=str(row["run_id"]),
            node_id=str(row["node_id"]),
            requested_at=float(row["requested_at"]),
            requested_by=str(row["requested_by"]),
            decision=decision,
            decided_at=decided_at,
            decided_by=principal.subject,
        )

    def recover_interrupted_runs(self, principal: Principal, *, updated_at: float) -> int:
        tenant = _tenant(principal)
        with self._write_lock, self._write() as connection:
            rows = connection.execute(
                """SELECT runs.run_id FROM workflow_runs AS runs
                JOIN workflows AS workflows ON workflows.workflow_id = runs.workflow_id
                WHERE workflows.tenant_id = ? AND runs.status = ?""",
                (tenant.value, WorkflowRunStatus.RUNNING.value),
            ).fetchall()
            run_ids = tuple(str(row["run_id"]) for row in rows)
            if not run_ids:
                return 0
            placeholders = ",".join("?" for _ in run_ids)
            connection.execute(
                f"UPDATE workflow_runs SET status = ?, updated_at = ?, error_code = ? WHERE run_id IN ({placeholders})",
                (WorkflowRunStatus.INTERRUPTED.value, updated_at, "runtime_interrupted", *run_ids),
            )
            connection.execute(
                f"""UPDATE workflow_step_runs SET status = ?, finished_at = ?, error_code = ?
                WHERE run_id IN ({placeholders}) AND status = ?""",
                (
                    WorkflowStepStatus.INTERRUPTED.value,
                    updated_at,
                    "runtime_interrupted",
                    *run_ids,
                    WorkflowStepStatus.RUNNING.value,
                ),
            )
        return len(run_ids)

    def _initialize(self) -> None:
        with self._write_lock, self._write() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 0:
                _create_schema(connection)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif version != _SCHEMA_VERSION:
                raise WorkflowStoreSchemaError()
            _validate_schema(connection)

    def _read(self) -> AbstractContextManager[sqlite3.Connection]:
        return self._database.read(WorkflowStoreStorageError)

    def _write(self) -> AbstractContextManager[sqlite3.Connection]:
        return self._database.immediate_transaction(
            WorkflowStoreStorageError,
            pass_through=(
                WorkflowStoreError,
                WorkflowModelError,
                WorkflowValidationError,
            ),
        )


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE workflows (
            workflow_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            display_name TEXT NOT NULL,
            active_revision_id TEXT,
            status TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX workflows_tenant_project_updated
            ON workflows (tenant_id, project_id, updated_at DESC, workflow_id DESC);
        CREATE TABLE workflow_drafts (
            workflow_id TEXT PRIMARY KEY REFERENCES workflows(workflow_id) ON DELETE RESTRICT,
            version INTEGER NOT NULL,
            specification_json TEXT,
            budget_json TEXT,
            updated_at REAL NOT NULL,
            updated_by TEXT NOT NULL,
            change_summary TEXT NOT NULL
        );
        CREATE TABLE workflow_revisions (
            revision_id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id) ON DELETE RESTRICT,
            revision_number INTEGER NOT NULL,
            specification_json TEXT NOT NULL,
            specification_digest TEXT NOT NULL,
            budget_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            created_by TEXT NOT NULL,
            change_summary TEXT NOT NULL,
            UNIQUE (workflow_id, revision_number)
        );
        CREATE TABLE workflow_deployments (
            deployment_id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id) ON DELETE RESTRICT,
            revision_id TEXT NOT NULL REFERENCES workflow_revisions(revision_id) ON DELETE RESTRICT,
            deployed_at REAL NOT NULL,
            deployed_by TEXT NOT NULL,
            status TEXT NOT NULL
        );
        CREATE UNIQUE INDEX workflow_deployments_one_active
            ON workflow_deployments (workflow_id) WHERE status = 'active';
        CREATE TABLE workflow_runs (
            run_id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id) ON DELETE RESTRICT,
            revision_id TEXT NOT NULL REFERENCES workflow_revisions(revision_id) ON DELETE RESTRICT,
            specification_digest TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            created_by TEXT NOT NULL,
            input_digest TEXT NOT NULL,
            error_code TEXT
        );
        CREATE INDEX workflow_runs_workflow_updated
            ON workflow_runs (workflow_id, updated_at DESC, run_id DESC);
        CREATE TABLE workflow_step_runs (
            step_run_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES workflow_runs(run_id) ON DELETE RESTRICT,
            node_id TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at REAL,
            finished_at REAL,
            input_digest TEXT,
            output_digest TEXT,
            error_code TEXT
        );
        CREATE INDEX workflow_step_runs_run ON workflow_step_runs (run_id, step_run_id);
        CREATE TABLE workflow_approvals (
            approval_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES workflow_runs(run_id) ON DELETE RESTRICT,
            node_id TEXT NOT NULL,
            requested_at REAL NOT NULL,
            requested_by TEXT NOT NULL,
            decision TEXT,
            decided_at REAL,
            decided_by TEXT
        );
        CREATE UNIQUE INDEX workflow_approvals_pending_node
            ON workflow_approvals (run_id, node_id) WHERE decision IS NULL;
        """
    )


def _validate_schema(connection: sqlite3.Connection) -> None:
    required = {
        "workflows",
        "workflow_drafts",
        "workflow_revisions",
        "workflow_deployments",
        "workflow_runs",
        "workflow_step_runs",
        "workflow_approvals",
    }
    actual = {
        str(row["name"])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    if not required <= actual:
        raise WorkflowStoreSchemaError()


def _workflow_from_row(row: sqlite3.Row) -> Workflow:
    try:
        return Workflow(
            workflow_id=str(row["workflow_id"]),
            tenant_id=TenantId(str(row["tenant_id"])),
            project_id=str(row["project_id"]),
            display_name=str(row["display_name"]),
            active_revision_id=row["active_revision_id"],
            status=WorkflowStatus(str(row["status"])),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise WorkflowStoreSchemaError() from error


def _draft_from_row(row: sqlite3.Row) -> WorkflowDraft:
    try:
        specification = _specification_from_json(row["specification_json"])
        budget = _budget_from_json(row["budget_json"])
        return WorkflowDraft(
            workflow_id=str(row["workflow_id"]),
            version=int(row["version"]),
            specification=specification,
            budget=budget,
            updated_at=float(row["updated_at"]),
            updated_by=str(row["updated_by"]),
            change_summary=str(row["change_summary"]),
        )
    except (TypeError, ValueError) as error:
        raise WorkflowStoreSchemaError() from error


def _revision_from_row(row: sqlite3.Row) -> WorkflowRevision:
    try:
        specification = _specification_from_json(row["specification_json"])
        budget = _budget_from_json(row["budget_json"])
        if specification is None or budget is None or specification.digest != str(row["specification_digest"]):
            raise ValueError
        return WorkflowRevision(
            revision_id=str(row["revision_id"]),
            workflow_id=str(row["workflow_id"]),
            revision_number=int(row["revision_number"]),
            specification=specification,
            budget=budget,
            created_at=float(row["created_at"]),
            created_by=str(row["created_by"]),
            change_summary=str(row["change_summary"]),
        )
    except (TypeError, ValueError) as error:
        raise WorkflowStoreSchemaError() from error


def _run_from_row(row: sqlite3.Row) -> WorkflowRun:
    try:
        return WorkflowRun(
            run_id=str(row["run_id"]),
            workflow_id=str(row["workflow_id"]),
            revision_id=str(row["revision_id"]),
            specification_digest=str(row["specification_digest"]),
            status=WorkflowRunStatus(str(row["status"])),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            created_by=str(row["created_by"]),
            input_digest=str(row["input_digest"]),
            error_code=row["error_code"],
        )
    except (TypeError, ValueError) as error:
        raise WorkflowStoreSchemaError() from error


def _step_from_row(row: sqlite3.Row) -> WorkflowStepRun:
    try:
        return WorkflowStepRun(
            step_run_id=str(row["step_run_id"]),
            run_id=str(row["run_id"]),
            node_id=str(row["node_id"]),
            status=WorkflowStepStatus(str(row["status"])),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            input_digest=row["input_digest"],
            output_digest=row["output_digest"],
            error_code=row["error_code"],
        )
    except (TypeError, ValueError) as error:
        raise WorkflowStoreSchemaError() from error


def _specification_json(specification: WorkflowSpec | None) -> str | None:
    return None if specification is None else specification.to_json()


def _specification_from_json(value: object) -> WorkflowSpec | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value.encode("utf-8")) > 256 * 1024:
        raise ValueError
    return WorkflowSpec.from_json(value)


def _budget_json(budget: ExecutionBudget | None) -> str | None:
    if budget is None:
        return None
    return json.dumps(
        {
            "max_steps": budget.max_steps,
            "max_model_calls": budget.max_model_calls,
            "max_wall_seconds": budget.max_wall_seconds,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _budget_from_json(value: object) -> ExecutionBudget | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError
    payload: Any = json.loads(value)
    if not isinstance(payload, dict) or set(payload) != {
        "max_steps",
        "max_model_calls",
        "max_wall_seconds",
    }:
        raise ValueError
    return ExecutionBudget(**payload)


def _tenant(principal: Principal) -> TenantId:
    if not isinstance(principal, Principal):
        raise WorkflowModelError("workflow principal is invalid")
    return principal.tenant_id


def _same_tenant(expected: TenantId, actual: TenantId) -> None:
    if expected != actual:
        raise WorkflowUnavailableError()


def _safe_workflow_id(value: object) -> str:
    try:
        return validate_workflow_id(value)
    except ValueError as error:
        raise WorkflowUnavailableError() from error


def _safe_revision_id(value: object) -> str:
    try:
        return validate_workflow_revision_id(value)
    except ValueError as error:
        raise WorkflowRevisionUnavailableError() from error


def _safe_run_id(value: object) -> str:
    try:
        return validate_workflow_run_id(value)
    except ValueError as error:
        raise WorkflowRunUnavailableError() from error


def _limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_LIST_LIMIT:
        raise WorkflowModelError("workflow list limit is invalid")
    return value


def _require_workflow(connection: sqlite3.Connection, tenant: TenantId, workflow_id: str) -> sqlite3.Row:
    row = connection.execute(
        "SELECT * FROM workflows WHERE workflow_id = ? AND tenant_id = ?", (workflow_id, tenant.value)
    ).fetchone()
    if row is None:
        raise WorkflowUnavailableError()
    return row


def _require_revision(
    connection: sqlite3.Connection, workflow_id: str, revision_id: str
) -> sqlite3.Row:
    row = connection.execute(
        "SELECT * FROM workflow_revisions WHERE workflow_id = ? AND revision_id = ?",
        (workflow_id, revision_id),
    ).fetchone()
    if row is None:
        raise WorkflowRevisionUnavailableError()
    return row


def _validate_run_transition(current: WorkflowRunStatus, target: WorkflowRunStatus) -> None:
    allowed = {
        WorkflowRunStatus.CREATED: {WorkflowRunStatus.QUEUED, WorkflowRunStatus.CANCELLED},
        WorkflowRunStatus.QUEUED: {
            WorkflowRunStatus.RUNNING,
            WorkflowRunStatus.CANCELLED,
            WorkflowRunStatus.INTERRUPTED,
        },
        WorkflowRunStatus.RUNNING: {
            WorkflowRunStatus.WAITING_APPROVAL,
            WorkflowRunStatus.SUCCEEDED,
            WorkflowRunStatus.FAILED,
            WorkflowRunStatus.CANCELLED,
            WorkflowRunStatus.INTERRUPTED,
        },
        WorkflowRunStatus.WAITING_APPROVAL: {
            WorkflowRunStatus.QUEUED,
            WorkflowRunStatus.FAILED,
            WorkflowRunStatus.CANCELLED,
        },
        WorkflowRunStatus.SUCCEEDED: set(),
        WorkflowRunStatus.FAILED: set(),
        WorkflowRunStatus.CANCELLED: set(),
        WorkflowRunStatus.INTERRUPTED: set(),
    }
    if target not in allowed[current]:
        raise WorkflowStoreStorageError()


__all__ = [
    "WorkflowApprovalUnavailableError",
    "WorkflowDraftConflictError",
    "WorkflowPublishConflictError",
    "WorkflowRevisionUnavailableError",
    "WorkflowRunUnavailableError",
    "WorkflowStore",
    "WorkflowStoreError",
    "WorkflowStoreSchemaError",
    "WorkflowStoreStorageError",
    "WorkflowUnavailableError",
]
