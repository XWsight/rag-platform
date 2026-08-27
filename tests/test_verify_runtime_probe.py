from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError

from scripts.verify_runtime_probe import RuntimeProbeError, read_api_key, run_probe, validate_base_url


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status


class RuntimeProbeTests(unittest.TestCase):
    def test_probe_uses_only_expected_read_only_paths_and_keeps_key_out_of_results(self) -> None:
        requests: list[object] = []

        def opener(request, timeout):
            self.assertEqual(timeout, 2.0)
            requests.append(request)
            return _Response(200)

        key = "probe-reader-key-0123456789"
        checks = run_probe("https://example.test", key, timeout_seconds=2.0, opener=opener)

        self.assertEqual(checks, ("live", "ready", "knowledge_bases"))
        self.assertEqual([request.get_method() for request in requests], ["GET", "GET", "GET"])
        self.assertEqual(
            [request.full_url for request in requests],
            [
                "https://example.test/health/live",
                "https://example.test/health/ready",
                "https://example.test/v1/knowledge-bases",
            ],
        )
        self.assertIsNone(requests[0].get_header("X-api-key"))
        self.assertEqual(requests[2].get_header("X-api-key"), key)
        self.assertNotIn(key, repr(checks))

    def test_probe_sanitizes_http_and_network_failures(self) -> None:
        def unavailable(request, timeout):
            raise HTTPError(request.full_url, 503, "private detail", {}, None)

        with self.assertRaisesRegex(RuntimeProbeError, "live probe returned HTTP 503"):
            run_probe("https://example.test", "probe-reader-key-0123456789", opener=unavailable)

    def test_url_and_key_file_boundaries_are_strict(self) -> None:
        with self.assertRaises(ValueError):
            validate_base_url("https://key@example.test")
        with self.assertRaises(ValueError):
            validate_base_url("https://example.test?secret=value")
        self.assertEqual(validate_base_url("https://example.test/base/"), "https://example.test/base")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "reader.token")
            path.write_text("probe-reader-key-0123456789\n", encoding="utf-8")
            self.assertEqual(read_api_key(path), "probe-reader-key-0123456789")
            path.write_text("short", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_api_key(path)


if __name__ == "__main__":
    unittest.main()
