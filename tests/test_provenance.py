from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_system.provenance import (
    SourceProvenance,
    inspect_source_provenance,
    require_clean_source,
)


class SourceProvenanceTests(unittest.TestCase):
    def test_inspection_validates_revision_and_records_clean_status(self) -> None:
        revision = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="a" * 40 + "\n",
            stderr="",
        )
        clean_status = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="",
        )
        with patch("rag_system.provenance._git", side_effect=(revision, clean_status)):
            provenance = inspect_source_provenance(Path("."))

        self.assertEqual(provenance.revision, "a" * 40)
        self.assertIs(provenance.working_tree_clean, True)

    def test_unknown_or_dirty_source_cannot_be_required_clean(self) -> None:
        with self.assertRaisesRegex(ValueError, "source revision"):
            require_clean_source(
                SourceProvenance(revision=None, working_tree_clean=True),
                artifact="a report",
            )
        with self.assertRaisesRegex(ValueError, "clean"):
            require_clean_source(
                SourceProvenance(revision="a" * 40, working_tree_clean=False),
                artifact="a report",
            )


if __name__ == "__main__":
    unittest.main()
