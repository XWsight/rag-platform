from __future__ import annotations

import unittest

from scripts.release_image_evidence import release_image_evidence


class ReleaseImageEvidenceTests(unittest.TestCase):
    def test_records_an_immutable_reference(self) -> None:
        digest = "sha256:" + "a" * 64
        evidence = release_image_evidence(
            image="ghcr.io/xwsight/rag-platform",
            digest=digest,
            source_revision="b" * 40,
        )
        self.assertEqual(evidence["immutable_reference"], f"ghcr.io/xwsight/rag-platform@{digest}")
        self.assertEqual(evidence["schema_version"], 1)

    def test_rejects_mutable_or_malformed_identifiers(self) -> None:
        with self.assertRaisesRegex(ValueError, "digest"):
            release_image_evidence(
                image="ghcr.io/xwsight/rag-platform",
                digest="latest",
                source_revision="b" * 40,
            )
        with self.assertRaisesRegex(ValueError, "revision"):
            release_image_evidence(
                image="ghcr.io/xwsight/rag-platform",
                digest="sha256:" + "a" * 64,
                source_revision="short",
            )


if __name__ == "__main__":
    unittest.main()
