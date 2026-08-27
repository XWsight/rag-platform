from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_system.sqlite_support import SqliteDatabase


class StorageFailure(RuntimeError):
    pass


class DomainFailure(RuntimeError):
    pass


class SqliteDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = SqliteDatabase(
            Path(self.directory.name, "state.sqlite3"),
            timeout_seconds=2.0,
        )

    def test_connect_enables_wal_foreign_keys_and_busy_timeout(self) -> None:
        connection = self.database.connect(StorageFailure)
        try:
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")
            self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            self.assertEqual(connection.execute("PRAGMA busy_timeout").fetchone()[0], 2_000)
        finally:
            connection.close()

    def test_connection_setup_failure_closes_connection_and_maps_error(self) -> None:
        connection = MagicMock()
        connection.execute.side_effect = sqlite3.OperationalError("disk unavailable")
        with patch("rag_system.sqlite_support.sqlite3.connect", return_value=connection):
            with self.assertRaises(StorageFailure):
                self.database.connect(StorageFailure)
        connection.close.assert_called_once_with()

    def test_read_maps_sql_errors_to_the_adapter_error(self) -> None:
        with self.assertRaises(StorageFailure):
            with self.database.read(StorageFailure) as connection:
                connection.execute("SELECT missing_column FROM missing_table")

    def test_transaction_commits_successful_changes(self) -> None:
        with closing(self.database.connect(StorageFailure)) as connection:
            connection.execute("CREATE TABLE entries (value TEXT NOT NULL)")

        with self.database.immediate_transaction(StorageFailure) as connection:
            connection.execute("INSERT INTO entries (value) VALUES ('committed')")

        with self.database.read(StorageFailure) as connection:
            values = connection.execute("SELECT value FROM entries").fetchall()
        self.assertEqual([row["value"] for row in values], ["committed"])

    def test_transaction_rolls_back_and_preserves_declared_domain_errors(self) -> None:
        with closing(self.database.connect(StorageFailure)) as connection:
            connection.execute("CREATE TABLE entries (value TEXT NOT NULL)")

        with self.assertRaises(DomainFailure):
            with self.database.immediate_transaction(
                StorageFailure,
                pass_through=(DomainFailure,),
            ) as connection:
                connection.execute("INSERT INTO entries (value) VALUES ('rolled-back')")
                raise DomainFailure("reject the transaction")

        with self.database.read(StorageFailure) as connection:
            count = connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
