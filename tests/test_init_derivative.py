from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from scripts.init_derivative import create_derivative, main


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
            provider_factory = (created / "provider_factory.py").read_text(encoding="utf-8")
            self.assertIn("class LegalAssistantProviderFactory", provider_factory)
            self.assertNotIn("{{", provider_factory)
            api_entrypoint = (created / "api_app.py").read_text(encoding="utf-8")
            self.assertIn("LegalAssistantProviderFactory", api_entrypoint)
            self.assertIn("RAG_PRODUCT_NAME=Legal Assistant", (created / ".env.example").read_text())

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


if __name__ == "__main__":
    unittest.main()
