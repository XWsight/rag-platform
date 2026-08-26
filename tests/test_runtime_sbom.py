from __future__ import annotations

import unittest

from scripts.generate_runtime_sbom import Distribution, SbomError, build_spdx


class RuntimeSbomTests(unittest.TestCase):
    def test_spdx_contains_the_runtime_dependency_closure_and_stable_edges(self) -> None:
        document = build_spdx(
            {"api": "1.0"},
            {
                "api": Distribution(
                    "API",
                    "1.0",
                    ("core>=2", "missing>=1", "dev-only; extra == 'dev'"),
                ),
                "core": Distribution("core", "2.0", ()),
            },
            project_name="Test Project",
        )

        self.assertEqual(document["spdxVersion"], "SPDX-2.3")
        self.assertEqual(
            [item["name"] for item in document["packages"]],
            ["API", "core"],
        )
        self.assertIn(
            {
                "spdxElementId": "SPDXRef-Package-api",
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": "SPDXRef-Package-core",
            },
            document["relationships"],
        )
        self.assertTrue(document["documentNamespace"].startswith("https://spdx.org/"))

    def test_declared_runtime_pin_must_match_the_installed_environment(self) -> None:
        with self.assertRaisesRegex(SbomError, "runtime requirements are empty"):
            build_spdx({}, {}, project_name="Test")
        with self.assertRaisesRegex(SbomError, "not installed"):
            build_spdx(
                {"api": "2.0"},
                {"api": Distribution("api", "1.0", ())},
                project_name="Test",
            )


if __name__ == "__main__":
    unittest.main()
