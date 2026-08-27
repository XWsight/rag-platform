"""Cross-repository startup contracts for durable SQLite adapters."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from rag_system.application_store import (
    ApplicationStore,
    ApplicationStoreSchemaError,
    ApplicationStoreStorageError,
)
from rag_system.catalog import CatalogSchemaError, CatalogStorageError, KnowledgeBaseCatalog
from rag_system.idempotency import (
    IdempotencySchemaError,
    IdempotencyStorageError,
    IdempotencyStore,
)
from rag_system.job_contracts import JobStorageError
from rag_system.job_store import SqliteJobSnapshotStore


class SqliteRepositoryStartupContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_connection_failures_are_mapped_to_public_storage_errors(self) -> None:
        repositories = (
            ("applications", lambda path: ApplicationStore(path), ApplicationStoreStorageError),
            ("catalog", lambda path: KnowledgeBaseCatalog(path), CatalogStorageError),
            ("idempotency", lambda path: IdempotencyStore(path), IdempotencyStorageError),
            ("job snapshots", lambda path: SqliteJobSnapshotStore(path), JobStorageError),
        )
        with patch(
            "rag_system.sqlite_support.sqlite3.connect",
            side_effect=sqlite3.OperationalError("injected connection failure"),
        ):
            for name, build, error_type in repositories:
                with self.subTest(repository=name):
                    with self.assertRaises(error_type):
                        build(Path(self.directory.name, f"{name}.sqlite3"))

    def test_unsupported_versions_are_rejected_without_being_rewritten(self) -> None:
        applications_path = Path(self.directory.name, "applications.sqlite3")
        ApplicationStore(applications_path)
        with closing(sqlite3.connect(applications_path)) as connection:
            connection.execute("PRAGMA user_version = 99")
            connection.commit()
        with self.assertRaises(ApplicationStoreSchemaError):
            ApplicationStore(applications_path)
        with closing(sqlite3.connect(applications_path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 99)

        catalog_path = Path(self.directory.name, "catalog.sqlite3")
        KnowledgeBaseCatalog(catalog_path)
        with closing(sqlite3.connect(catalog_path)) as connection:
            connection.execute("PRAGMA user_version = 99")
            connection.commit()
        with self.assertRaises(CatalogSchemaError):
            KnowledgeBaseCatalog(catalog_path)
        with closing(sqlite3.connect(catalog_path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 99)

        idempotency_path = Path(self.directory.name, "idempotency.sqlite3")
        IdempotencyStore(idempotency_path)
        with closing(sqlite3.connect(idempotency_path)) as connection:
            connection.execute("UPDATE idempotency_meta SET schema_version = 99")
            connection.commit()
        with self.assertRaises(IdempotencySchemaError):
            IdempotencyStore(idempotency_path)
        with closing(sqlite3.connect(idempotency_path)) as connection:
            version = connection.execute(
                "SELECT schema_version FROM idempotency_meta"
            ).fetchone()[0]
        self.assertEqual(version, 99)

        jobs_path = Path(self.directory.name, "jobs.sqlite3")
        SqliteJobSnapshotStore(jobs_path)
        with closing(sqlite3.connect(jobs_path)) as connection:
            connection.execute("PRAGMA user_version = 99")
            connection.commit()
        with self.assertRaises(JobStorageError):
            SqliteJobSnapshotStore(jobs_path)
        with closing(sqlite3.connect(jobs_path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 99)


if __name__ == "__main__":
    unittest.main()
