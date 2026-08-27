"""Run privacy-safe, read-only checks against one deployed API instance."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


class RuntimeProbeError(RuntimeError):
    """A sanitized failure from a deployment verification request."""


UrlOpener = Callable[[Request, float], Any]


def _open_request(request: Request, timeout_seconds: float) -> Any:
    return urlopen(request, timeout=timeout_seconds)


def validate_base_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base URL must be an absolute HTTP(S) address without credentials")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def read_api_key(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError("API key file must be a regular file")
    value = path.read_text(encoding="utf-8").strip()
    if len(value) < 16 or len(value) > 4_096:
        raise ValueError("API key file has an invalid length")
    return value


def run_probe(
    base_url: str,
    api_key: str,
    *,
    timeout_seconds: float = 10.0,
    opener: UrlOpener = _open_request,
) -> tuple[str, ...]:
    if len(api_key) < 16:
        raise ValueError("API key has an invalid length")
    if timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    checks = (
        ("live", "/health/live", False),
        ("ready", "/health/ready", False),
        ("knowledge_bases", "/v1/knowledge-bases", True),
    )
    passed: list[str] = []
    for label, path, authenticated in checks:
        headers = {"Accept": "application/json"}
        if authenticated:
            headers["X-API-Key"] = api_key
        request = Request(f"{base_url}{path}", headers=headers, method="GET")
        try:
            with opener(request, timeout_seconds) as response:
                status = int(response.getcode())
        except HTTPError as error:
            status = error.code
        except (OSError, URLError):
            raise RuntimeProbeError(f"{label} probe is unavailable") from None
        if status != 200:
            raise RuntimeProbeError(f"{label} probe returned HTTP {status}")
        passed.append(label)
    return tuple(passed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-file", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    try:
        base_url = validate_base_url(args.base_url)
        api_key = read_api_key(args.api_key_file)
        checks = run_probe(base_url, api_key, timeout_seconds=args.timeout_seconds)
    except (OSError, RuntimeProbeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"base_url": base_url, "checks": checks, "status": "ok"}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
