from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_system.application import PlatformUnavailableError, UploadDocument
from rag_system.knowledge_base_assets import KnowledgeBaseAssets
from rag_system.file_store import TenantFileStore
from rag_system.tenancy import Principal, TenantId


class KnowledgeBaseAssetsTests(unittest.TestCase):
    def test_plan_rejects_duplicate_document_identifiers_before_storing_files(self) -> None:
        principal = Principal("writer", TenantId("tenant"), frozenset({"writer"}))
        with tempfile.TemporaryDirectory() as directory:
            assets = KnowledgeBaseAssets(TenantFileStore(Path(directory) / "documents"))

            with self.assertRaises(PlatformUnavailableError):
                assets.plan(
                    principal,
                    (
                        UploadDocument("first.md", b"first"),
                        UploadDocument("second.md", b"second"),
                    ),
                    new_document_id=lambda: "doc_0123456789abcdef",
                )

            self.assertEqual(tuple((Path(directory) / "documents").iterdir()), ())


if __name__ == "__main__":
    unittest.main()
