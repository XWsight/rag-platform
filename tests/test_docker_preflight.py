from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.docker_preflight import (
    DockerPreflightError,
    verify_compose_prerequisites,
    verify_docker_engine,
)


def _result(*, code: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(("docker",), code, stdout=stdout, stderr=stderr)


class DockerPreflightTests(unittest.TestCase):
    def test_engine_reports_server_version(self) -> None:
        self.assertEqual(verify_docker_engine(run=lambda _: _result(code=0, stdout="29.7.2\n")), "29.7.2")

    def test_missing_desktop_engine_has_actionable_failure(self) -> None:
        with self.assertRaisesRegex(DockerPreflightError, "Linux engine is unavailable"):
            verify_docker_engine(
                run=lambda _: _result(code=1, stderr="open dockerDesktopLinuxEngine: missing")
            )

    def test_compose_requires_a_regular_local_environment_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DockerPreflightError, "regular .env"):
                verify_compose_prerequisites(root=Path(directory), run=lambda _: _result(code=0))

    def test_compose_rejects_invalid_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("RAG_PERSIST_DATA=true\n", encoding="utf-8")
            with self.assertRaisesRegex(DockerPreflightError, "configuration is invalid"):
                verify_compose_prerequisites(root=root, run=lambda _: _result(code=1))


if __name__ == "__main__":
    unittest.main()
