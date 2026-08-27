"""Generate and verify the reviewed public OpenAPI v1 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.api import create_app  # noqa: E402
from rag_system.application import RagApplication  # noqa: E402
from rag_system.config import Settings  # noqa: E402
from rag_system.observability import JsonEventLogger  # noqa: E402
from rag_system.rate_limit import TokenBucketRateLimiter  # noqa: E402
from rag_system.tenancy import ApiKeyAuthenticator, Principal, TenantId  # noqa: E402


DEFAULT_CONTRACT = PROJECT_ROOT / "contracts" / "openapi-v1.json"


class OpenApiContractError(ValueError):
    """The reviewed OpenAPI contract is absent, malformed, or has drifted."""


@dataclass(frozen=True, slots=True)
class OpenApiContractSummary:
    """Non-sensitive evidence about the reviewed HTTP contract."""

    endpoint_count: int
    fingerprint: str


class _SchemaPlatform:
    """Minimal composition dependency used only to expose route metadata."""

    def __init__(self) -> None:
        self.settings = Settings()

    def close(self) -> None:
        """Match the application lifecycle port without allocating runtime resources."""


def build_contract() -> dict[str, Any]:
    """Build a deterministic schema without opening storage, models, or network connections."""

    principal = Principal(
        "openapi-contract",
        TenantId("contract"),
        frozenset({"reader", "writer", "operator"}),
    )
    logger = logging.Logger("rag-platform.openapi-contract")
    logger.addHandler(logging.NullHandler())
    app = create_app(
        platform=cast(RagApplication, _SchemaPlatform()),
        authenticator=ApiKeyAuthenticator.from_mapping(
            {"openapi-contract-key-0123456789": principal}
        ),
        rate_limiter=TokenBucketRateLimiter(rate_per_second=100.0, capacity=100.0),
        logger=JsonEventLogger(logger),
        close_on_shutdown=False,
    )
    schema = json.loads(json.dumps(app.openapi()))
    if not isinstance(schema, dict):
        raise OpenApiContractError("OpenAPI generator returned an invalid document")
    info = schema.get("info")
    if isinstance(info, dict):
        # Package releases change this display-only value without changing the
        # v1 wire contract; preserve all schema and endpoint metadata instead.
        info.pop("version", None)
    return {"schema_version": 1, "openapi": schema}


def write_contract(path: Path = DEFAULT_CONTRACT) -> OpenApiContractSummary:
    """Write the currently generated contract for explicit review in a PR."""

    destination = path.resolve()
    rendered = _render(build_contract())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    return _summary(rendered)


def verify_contract(path: Path = DEFAULT_CONTRACT) -> OpenApiContractSummary:
    """Fail closed when the generated v1 wire contract differs from its baseline."""

    expected = _load_contract(path.resolve())
    actual = build_contract()
    if expected != actual:
        raise OpenApiContractError(
            "OpenAPI contract drift detected; review the API compatibility impact and run "
            "scripts/verify_openapi_contract.py --update only for an intentional v1 change"
        )
    return _summary(_render(actual))


def _load_contract(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OpenApiContractError("OpenAPI contract cannot be read") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise OpenApiContractError("OpenAPI contract schema is unsupported")
    if not isinstance(payload.get("openapi"), dict):
        raise OpenApiContractError("OpenAPI contract document is invalid")
    return cast(dict[str, Any], payload)


def _render(contract: dict[str, Any]) -> str:
    return json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _summary(rendered: str) -> OpenApiContractSummary:
    payload = json.loads(rendered)
    openapi = payload["openapi"]
    return OpenApiContractSummary(
        endpoint_count=len(openapi["paths"]),
        fingerprint=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--update",
        action="store_true",
        help="Write a reviewed baseline after an intentional API compatibility review.",
    )
    arguments = parser.parse_args(argv)
    try:
        summary = write_contract(arguments.contract) if arguments.update else verify_contract(arguments.contract)
    except OpenApiContractError as error:
        parser.error(str(error))
    action = "updated" if arguments.update else "verified"
    print(
        f"OpenAPI v1 contract {action}: {summary.endpoint_count} paths, "
        f"sha256={summary.fingerprint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
