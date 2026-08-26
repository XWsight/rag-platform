from __future__ import annotations

import unittest

from rag_system.catalog import (
    DocumentManifest as LegacyDocumentManifest,
    KnowledgeBaseErrorCode as LegacyKnowledgeBaseErrorCode,
    KnowledgeBaseRecord as LegacyKnowledgeBaseRecord,
    KnowledgeBaseStatus as LegacyKnowledgeBaseStatus,
)
from rag_system.knowledge_base_contracts import (
    DocumentManifest,
    KnowledgeBaseErrorCode,
    KnowledgeBaseRecord,
    KnowledgeBaseStatus,
)


class KnowledgeBaseContractCompatibilityTests(unittest.TestCase):
    def test_catalog_reexports_storage_neutral_contract_types(self) -> None:
        self.assertIs(LegacyDocumentManifest, DocumentManifest)
        self.assertIs(LegacyKnowledgeBaseErrorCode, KnowledgeBaseErrorCode)
        self.assertIs(LegacyKnowledgeBaseRecord, KnowledgeBaseRecord)
        self.assertIs(LegacyKnowledgeBaseStatus, KnowledgeBaseStatus)


if __name__ == "__main__":
    unittest.main()
