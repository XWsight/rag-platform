"""Verify one published application through read-only production API calls."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.application_contracts import validate_application_id  # noqa: E402
from scripts.verify_runtime_probe import read_api_key, validate_base_url  # noqa: E402


class ApplicationProbeError(RuntimeError):
    """A sanitized application publication verification failure."""


UrlOpener = Callable[[Request, float], Any]


def _open_request(request: Request, timeout_seconds: float) -> Any:
    return urlopen(request, timeout=timeout_seconds)


def run_application_probe(
    base_url: str,
    api_key: str,
    application_id: str,
    *,
    timeout_seconds: float = 10.0,
    opener: UrlOpener = _open_request,
) -> tuple[str, ...]:
    validate_application_id(application_id)
    if len(api_key) < 16:
        raise ValueError("API key has an invalid length")
    if timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    root = f"{base_url}/v1/applications/{quote(application_id, safe='')}"
    payloads: dict[str, dict[str, Any]] = {}
    for label, path in (
        ("application", root),
        ("revisions", f"{root}/revisions"),
        ("deployments", f"{root}/deployments"),
    ):
        request = Request(
            path, headers={"Accept": "application/json", "X-API-Key": api_key}, method="GET"
        )
        try:
            with opener(request, timeout_seconds) as response:
                if int(response.getcode()) != 200:
                    raise ApplicationProbeError(f"{label} probe returned a non-success status")
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise ApplicationProbeError(f"{label} probe returned HTTP {error.code}") from None
        except (OSError, UnicodeError, URLError, json.JSONDecodeError, AttributeError):
            raise ApplicationProbeError(f"{label} probe is unavailable") from None
        if not isinstance(payload, dict):
            raise ApplicationProbeError(f"{label} probe returned an invalid contract")
        payloads[label] = payload
    active_revision = payloads["application"].get("active_revision_id")
    revisions = payloads["revisions"].get("items")
    deployments = payloads["deployments"].get("items")
    if (
        not isinstance(active_revision, str)
        or not isinstance(revisions, list)
        or not any(item.get("id") == active_revision for item in revisions if isinstance(item, dict))
        or not isinstance(deployments, list)
        or not any(
            item.get("revision_id") == active_revision
            for item in deployments
            if isinstance(item, dict)
        )
    ):
        raise ApplicationProbeError("published application state is inconsistent")
    return "application", "revisions", "deployments", "publication"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-file", required=True, type=Path)
    parser.add_argument("--application-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    try:
        base_url = validate_base_url(args.base_url)
        checks = run_application_probe(
            base_url, read_api_key(args.api_key_file), args.application_id,
            timeout_seconds=args.timeout_seconds,
        )
    except (ApplicationProbeError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"application_id": args.application_id, "checks": checks, "status": "ok"}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
