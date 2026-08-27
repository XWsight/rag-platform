"""Small, adapter-only SQLite connection and transaction primitives."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar


StorageError = TypeVar("StorageError", bound=Exception)


class SqliteDatabase:
    """Open short-lived SQLite connections with one explicit durability policy.

    Adapters retain ownership of schemas, validation, and public error types.
    This helper only centralizes connection setup and ``BEGIN IMMEDIATE``.
    """

    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float,
        require_wal: bool = True,
        synchronous_normal: bool = False,
    ) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.require_wal = require_wal
        self.synchronous_normal = synchronous_normal

    def connect(self, storage_error: Callable[[], StorageError]) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self.path,
                timeout=self.timeout_seconds,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}")
            mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
            if self.require_wal and mode != "wal":
                raise _ConnectionSetupError()
            if self.synchronous_normal:
                connection.execute("PRAGMA synchronous = NORMAL")
            return connection
        except (_ConnectionSetupError, sqlite3.Error) as error:
            if connection is not None:
                connection.close()
            raise storage_error() from error

    @contextmanager
    def read(self, storage_error: Callable[[], StorageError]) -> Iterator[sqlite3.Connection]:
        connection = self.connect(storage_error)
        try:
            yield connection
        except sqlite3.Error as error:
            raise storage_error() from error
        finally:
            connection.close()

    @contextmanager
    def immediate_transaction(
        self,
        storage_error: Callable[[], StorageError],
        *,
        pass_through: tuple[type[Exception], ...] = (),
        before_write: Callable[[sqlite3.Connection], None] | None = None,
    ) -> Iterator[sqlite3.Connection]:
        connection = self.connect(storage_error)
        try:
            connection.execute("BEGIN IMMEDIATE")
            if before_write is not None:
                before_write(connection)
            yield connection
            connection.commit()
        except pass_through:
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise storage_error() from error
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


class _ConnectionSetupError(RuntimeError):
    """Internal sentinel for a SQLite connection that cannot meet its policy."""


__all__ = ["SqliteDatabase"]
