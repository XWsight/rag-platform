from __future__ import annotations

import json
import unittest

from scripts.verify_application_probe import ApplicationProbeError, run_application_probe


APPLICATION_ID = "app_0123456789abcdef0123456789abcdef"
REVISION_ID = "rev_0123456789abcdef0123456789abcdef"


class Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def getcode(self) -> int:
        return 200

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class ApplicationProbeTests(unittest.TestCase):
    def test_probe_is_read_only_and_requires_consistent_publication(self) -> None:
        requests = []
        payloads = iter((
            {"active_revision_id": REVISION_ID},
            {"items": [{"id": REVISION_ID}]},
            {"items": [{"revision_id": REVISION_ID}]},
        ))

        def opener(request, timeout):
            self.assertEqual(timeout, 2.0)
            requests.append(request)
            return Response(next(payloads))

        checks = run_application_probe(
            "https://example.test", "reader-key-0123456789", APPLICATION_ID,
            timeout_seconds=2.0, opener=opener,
        )
        self.assertEqual(checks, ("application", "revisions", "deployments", "publication"))
        self.assertEqual({request.get_method() for request in requests}, {"GET"})
        self.assertTrue(all(request.get_header("X-api-key") for request in requests))

    def test_probe_rejects_inconsistent_publication_without_leaking_payload(self) -> None:
        payloads = iter((
            {"active_revision_id": REVISION_ID}, {"items": []}, {"items": []},
        ))
        with self.assertRaisesRegex(ApplicationProbeError, "state is inconsistent"):
            run_application_probe(
                "https://example.test", "reader-key-0123456789", APPLICATION_ID,
                opener=lambda request, timeout: Response(next(payloads)),
            )


if __name__ == "__main__":
    unittest.main()
