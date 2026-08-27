from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.migrate_volume import VolumeMigrationError, build_plan, execute_plan


class VolumeMigrationPlanTests(unittest.TestCase):
    def test_plan_is_non_destructive_until_explicitly_executed(self) -> None:
        plan = build_plan(source_volume="source-data", destination_volume="rag-platform-data")

        self.assertEqual(plan.create_command, ("docker", "volume", "create", "rag-platform-data"))
        self.assertIn("source-data:/source:ro", plan.copy_command)
        self.assertIn("rag-platform-data:/destination", plan.copy_command)
        self.assertIn("tar -cf - . | tar -C /destination -xf -", plan.copy_command[-1])
        self.assertIn("sha256sum", plan.manifest_command("rag-platform-data")[-1])

    def test_plan_rejects_unsafe_or_overwriting_volume_names(self) -> None:
        for source, destination in (
            ("source-data", "source-data"),
            ("source-data", "other:/data"),
            ("../../source", "rag-platform-data"),
        ):
            with self.subTest(source=source, destination=destination):
                with self.assertRaises(VolumeMigrationError):
                    build_plan(source_volume=source, destination_volume=destination)

    def test_execute_preserves_the_source_and_reports_a_verified_copy(self) -> None:
        plan = build_plan(source_volume="source-data", destination_volume="rag-platform-data")
        manifest = "abc  ./catalog.sqlite3\ndef  ./documents/file.md\n"
        with patch("scripts.migrate_volume._require_volume") as require, patch(
            "scripts.migrate_volume._run", side_effect=("", "", manifest, manifest)
        ) as run:
            result = execute_plan(plan)

        require.assert_any_call("source-data", must_exist=True)
        require.assert_any_call("rag-platform-data", must_exist=False)
        self.assertEqual(run.call_count, 4)
        self.assertEqual(result["file_count"], 2)
        self.assertTrue(result["verified"])

    def test_execute_keeps_destination_for_manual_inspection_after_a_mismatch(self) -> None:
        plan = build_plan(source_volume="source-data", destination_volume="rag-platform-data")
        with patch("scripts.migrate_volume._require_volume"), patch(
            "scripts.migrate_volume._run",
            side_effect=("", "", "abc  ./catalog.sqlite3\n", "def  ./catalog.sqlite3\n"),
        ):
            with self.assertRaisesRegex(VolumeMigrationError, "destination volume was preserved"):
                execute_plan(plan)


if __name__ == "__main__":
    unittest.main()
