from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from scripts.init_derivative import create_derivative, main
from scripts.validate_derivative_compatibility import (
    DerivativeCompatibilityError,
    validate_compatibility,
)


class InitDerivativeTests(unittest.TestCase):
    def test_creates_a_rendered_derivative_layer_without_touching_the_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "legal_assistant"

            created = create_derivative(
                package_name="legal_assistant",
                output=destination,
                product_name="Legal Assistant",
                product_tagline="Evidence-first legal workspace",
            )

            self.assertEqual(created, destination.resolve())
            self.assertTrue((created / "README.md").is_file())
            compatibility = created / "compatibility.json"
            self.assertTrue(compatibility.is_file())
            self.assertTrue(validate_compatibility(compatibility, base_root=Path.cwd())["compatible"])
            provider_factory = (created / "provider_factory.py").read_text(encoding="utf-8")
            self.assertIn("class LegalAssistantProviderFactory", provider_factory)
            self.assertNotIn("{{", provider_factory)
            api_entrypoint = (created / "api_app.py").read_text(encoding="utf-8")
            self.assertIn("LegalAssistantProviderFactory", api_entrypoint)
            self.assertIn("RAG_PRODUCT_NAME=Legal Assistant", (created / ".env.example").read_text())
            self.assertIn("Upstream baseline", (created / "UPSTREAM.md").read_text())
            self.assertNotIn("{{", (created / "UPSTREAM.md").read_text())
            governance = json.loads((created / "evals" / "governance.json").read_text())
            self.assertEqual(governance["status"], "draft")
            self.assertEqual(governance["product_name"], "Legal Assistant")
            self.assertNotIn("{{", json.dumps(governance))
            workflow = (
                created / ".github" / "workflows" / "derivative-evaluation.yml"
            ).read_text(encoding="utf-8")
            self.assertIn("--require-ready", workflow)
            self.assertIn("legal_assistant/evals/governance.json", workflow)
            self.assertNotIn("{{", workflow)

    def test_compatibility_manifest_rejects_an_incompatible_api_major(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "compatibility.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "base_project": "rag-studio",
                        "base_revision": "0123456",
                        "base_api_major": 3,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(DerivativeCompatibilityError):
                validate_compatibility(manifest, base_root=Path.cwd())

    def test_accepts_an_explicit_base_revision_for_a_reproducible_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            revision = "0123456789abcdef0123456789abcdef01234567"
            created = create_derivative(
                package_name="research_assistant",
                output=Path(directory) / "research_assistant",
                product_name="Research Assistant",
                product_tagline="Evidence workspace",
                base_revision=revision,
            )
            self.assertIn(revision, (created / "UPSTREAM.md").read_text(encoding="utf-8"))

            with self.assertRaises(ValueError):
                create_derivative(
                    package_name="invalid_revision",
                    output=Path(directory) / "invalid_revision",
                    product_name="Invalid Revision",
                    product_tagline="Evidence workspace",
                    base_revision="not-a-git-revision",
                )

    def test_refuses_to_overwrite_an_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "existing"
            destination.mkdir()

            with self.assertRaises(FileExistsError):
                create_derivative(
                    package_name="existing_layer",
                    output=destination,
                    product_name="Existing Layer",
                    product_tagline="Evidence workspace",
                )

    def test_render_failure_leaves_no_partial_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "partial"
            with patch("scripts.init_derivative.Path.write_text", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    create_derivative(
                        package_name="partial_layer",
                        output=destination,
                        product_name="Partial Layer",
                        product_tagline="Evidence workspace",
                    )
            self.assertFalse(destination.exists())

    def test_command_reports_invalid_inputs_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "invalid"

            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(
                    [
                        "--package-name",
                        "Invalid-Name",
                        "--output",
                        str(destination),
                        "--product-name",
                        "Invalid",
                        "--product-tagline",
                        "Evidence workspace",
                    ]
                )

            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(destination.exists())

    def test_generated_api_assembles_and_serves_an_authenticated_request(self) -> None:
        package_name = "legal_assistant"
        api_key = "derivative-key-0123456789abcdef"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_derivative(
                package_name=package_name,
                output=root / package_name,
                product_name="Legal Assistant",
                product_tagline="Evidence-first legal workspace",
            )
            environment = {
                "RAG_PERSIST_DATA": "true",
                "RAG_STORAGE_ROOT": str(root / "data"),
                "RAG_PRODUCT_NAME": "Legal Assistant",
                "RAG_PRODUCT_TAGLINE": "Evidence-first legal workspace",
                "RAG_API_KEYS_JSON": json.dumps(
                    {
                        api_key: {
                            "subject": "derivative-test",
                            "tenant_id": "derivative",
                            "roles": ["reader", "writer", "operator"],
                        }
                    }
                ),
            }
            sys.path.insert(0, str(root))
            try:
                with patch.dict(os.environ, environment, clear=True):
                    with patch("logging.basicConfig"):
                        module = importlib.import_module(f"{package_name}.api_app")
                    with TestClient(module.app, raise_server_exceptions=False) as client:
                        self.assertEqual(client.get("/health/ready").status_code, 200)
                        response = client.get(
                            "/v1/knowledge-bases",
                            headers={"X-API-Key": api_key},
                        )
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.json()["items"], [])
                        self.assertEqual(client.get("/openapi.json").json()["info"]["title"], "Legal Assistant API")
                    provider_test = importlib.import_module(
                        f"{package_name}.tests.test_provider_factory"
                    )
                    result = unittest.TextTestRunner(stream=io.StringIO()).run(
                        unittest.defaultTestLoader.loadTestsFromModule(provider_test)
                    )
                    self.assertTrue(result.wasSuccessful())
            finally:
                sys.path.remove(str(root))
                for module_name in tuple(sys.modules):
                    if module_name == package_name or module_name.startswith(f"{package_name}."):
                        del sys.modules[module_name]


if __name__ == "__main__":
    unittest.main()
