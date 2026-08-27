from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_openapi_contract import (
    OpenApiContractError,
    build_contract,
    verify_contract,
    write_contract,
)


class OpenApiContractTests(unittest.TestCase):
    def test_contract_can_be_written_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contract_path = Path(directory) / "openapi-v1.json"
            written = write_contract(contract_path)
            verified = verify_contract(contract_path)

        self.assertGreater(written.endpoint_count, 0)
        self.assertEqual(verified, written)

    def test_contract_ignores_package_display_version(self) -> None:
        document = build_contract()
        self.assertNotIn("version", document["openapi"]["info"])

    def test_contract_rejects_unreviewed_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contract_path = Path(directory) / "openapi-v1.json"
            write_contract(contract_path)
            document = json.loads(contract_path.read_text(encoding="utf-8"))
            document["openapi"]["paths"] = {}
            contract_path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(OpenApiContractError, "drift"):
                verify_contract(contract_path)
