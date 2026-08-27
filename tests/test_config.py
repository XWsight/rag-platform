from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rag_system.config import SecretValue, Settings


class ConfigurationTests(unittest.TestCase):
    def test_secret_never_appears_in_string_representation(self) -> None:
        secret = SecretValue("very-private-value")
        self.assertNotIn("very-private-value", repr(secret))
        self.assertNotIn("very-private-value", str(secret))
        self.assertEqual(secret.reveal(), "very-private-value")

    def test_secret_file_source_is_loaded_without_retaining_its_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secret_path = Path(directory) / "api-keys.json"
            secret_path.write_text("from-secret-file\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"RAG_API_KEYS_JSON_FILE": str(secret_path)},
                clear=True,
            ):
                settings = Settings()

        self.assertEqual(settings.api_keys_json.reveal(), "from-secret-file")
        self.assertNotIn("api-keys.json", repr(settings.api_keys_json))

    def test_secret_environment_source_remains_backward_compatible(self) -> None:
        with patch.dict(os.environ, {"ZHIPU_API_KEY": "from-environment"}, clear=True):
            settings = Settings()

        self.assertEqual(settings.api_key.reveal(), "from-environment")

    def test_secret_file_source_rejects_ambiguous_or_unsafe_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secret_path = Path(directory) / "provider-key"
            secret_path.write_text("from-secret-file", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "ZHIPU_API_KEY": "environment-value",
                    "ZHIPU_API_KEY_FILE": str(secret_path),
                },
                clear=True,
            ), self.assertRaisesRegex(ValueError, "cannot both be set"):
                Settings()

            with patch.dict(
                os.environ,
                {"ZHIPU_API_KEY_FILE": "relative-secret"},
                clear=True,
            ), self.assertRaisesRegex(ValueError, "absolute path"):
                Settings()

            with patch.dict(
                os.environ,
                {"ZHIPU_API_KEY_FILE": str(Path(directory) / "missing")},
                clear=True,
            ), self.assertRaisesRegex(ValueError, "regular non-symbolic file"):
                Settings()

            oversized_path = Path(directory) / "oversized"
            oversized_path.write_bytes(b"x" * (64 * 1024 + 1))
            with patch.dict(
                os.environ,
                {"ZHIPU_API_KEY_FILE": str(oversized_path)},
                clear=True,
            ), self.assertRaisesRegex(ValueError, "invalid content"):
                Settings()

    def test_compose_secret_overlay_removes_plaintext_credential_values(self) -> None:
        overlay = (Path(__file__).resolve().parents[1] / "compose.secrets.example.yaml").read_text(
            encoding="utf-8"
        )

        self.assertIn("RAG_API_KEYS_JSON: \"\"", overlay)
        self.assertIn("RAG_API_KEYS_JSON_FILE: /run/secrets/rag_api_keys_json", overlay)
        self.assertIn("ZHIPU_API_KEY: \"\"", overlay)
        self.assertIn("ZHIPU_API_KEY_FILE: /run/secrets/zhipu_api_key", overlay)
        self.assertIn("file: ./secrets/rag_api_keys_json", overlay)
        self.assertIn("file: ./secrets/zhipu_api_key", overlay)

    def test_defaults_are_valid_and_privacy_preserving(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings().validate()
        self.assertFalse(settings.allow_cloud_default)
        self.assertFalse(settings.allow_web_default)
        self.assertFalse(settings.persist_data)
        self.assertGreater(settings.chunk_size, settings.chunk_overlap)
        self.assertGreaterEqual(settings.max_jobs, settings.job_workers)
        self.assertGreater(settings.job_history_ttl_seconds, settings.job_ttl_seconds)
        self.assertGreaterEqual(settings.job_history_max_per_tenant, settings.max_jobs)
        self.assertEqual(settings.product_name, "RAG Platform")

    def test_invalid_chunking_and_url_are_rejected(self) -> None:
        settings = Settings()
        with self.assertRaises(ValueError):
            replace(settings, chunk_overlap=settings.chunk_size).validate()
        with self.assertRaises(ValueError):
            replace(settings, chat_url="http://example.com/chat").validate()
        for field_name, value in (
            ("chat_url", "https://example.com:bad/chat"),
            ("search_url", "https://example.com:99999/search"),
            ("chat_url", "https://user:pass@example.com/chat"),
            ("search_url", "https://example.com:0/search"),
        ):
            with self.subTest(field_name=field_name, value=value), self.assertRaises(ValueError):
                replace(settings, **{field_name: value}).validate()

    def test_non_finite_timeouts_and_character_bounds_are_rejected(self) -> None:
        settings = Settings()
        for timeout in (float("nan"), float("inf"), 0.0, 301.0):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                replace(settings, read_timeout_seconds=timeout).validate()
        with self.assertRaises(ValueError):
            replace(settings, max_question_characters=0).validate()
        with self.assertRaises(ValueError):
            replace(settings, max_context_characters=999).validate()
        with self.assertRaises(ValueError):
            replace(settings, answer_max_tokens=511).validate()
        with self.assertRaises(ValueError):
            replace(settings, query_plan_max_tokens=4_097).validate()
        with self.assertRaises(ValueError):
            replace(settings, max_concurrent_answers=0).validate()
        with self.assertRaises(ValueError):
            replace(settings, job_history_ttl_seconds=59).validate()
        with self.assertRaises(ValueError):
            replace(
                settings,
                job_history_ttl_seconds=settings.job_ttl_seconds - 1,
            ).validate()
        with self.assertRaises(ValueError):
            replace(settings, job_history_max_per_tenant=0).validate()
        with self.assertRaises(ValueError):
            replace(
                settings,
                job_history_max_per_tenant=settings.max_jobs_per_tenant - 1,
            ).validate()
        for ratio in (0.49, 1.0, float("nan")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                replace(settings, hybrid_confidence_ratio=ratio).validate()
        for saturation in (0.04, 1.01, float("inf")):
            with self.subTest(saturation=saturation), self.assertRaises(ValueError):
                replace(settings, routing_lexical_saturation=saturation).validate()
        for minimum in (-0.01, 1.01, float("nan")):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                replace(settings, routing_min_lexical_score=minimum).validate()
        for field_name, value in (
            ("product_name", ""),
            ("product_name", "name\nwith-newline"),
            ("product_tagline", "x" * 161),
        ):
            with self.subTest(field_name=field_name, value=value), self.assertRaises(ValueError):
                replace(settings, **{field_name: value}).validate()


if __name__ == "__main__":
    unittest.main()
