"""Bounded JSON normalization for asynchronous job results."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping


class InvalidJobResultError(ValueError):
    """Raised when a task result cannot be stored under the executor contract."""


def canonical_job_result(
    value: object,
    *,
    max_bytes: int,
    max_depth: int,
    max_items: int,
) -> str:
    if not isinstance(value, Mapping):
        raise InvalidJobResultError()
    item_counter = [0]
    normalized = _normalize_json_value(
        value,
        depth=0,
        max_depth=max_depth,
        max_items=max_items,
        item_counter=item_counter,
        active_ids=set(),
    )
    try:
        payload = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError):
        raise InvalidJobResultError() from None
    if len(payload.encode("utf-8")) > max_bytes:
        raise InvalidJobResultError()
    return payload


def _normalize_json_value(
    value: object,
    *,
    depth: int,
    max_depth: int,
    max_items: int,
    item_counter: list[int],
    active_ids: set[int],
) -> object:
    if depth > max_depth:
        raise InvalidJobResultError()
    item_counter[0] += 1
    if item_counter[0] > max_items:
        raise InvalidJobResultError()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidJobResultError()
        return value
    if isinstance(value, (Mapping, list, tuple)):
        identity = id(value)
        if identity in active_ids:
            raise InvalidJobResultError()
        active_ids.add(identity)
        try:
            if isinstance(value, Mapping):
                normalized_mapping: dict[str, object] = {}
                for key, child in value.items():
                    if not isinstance(key, str):
                        raise InvalidJobResultError()
                    normalized_mapping[key] = _normalize_json_value(
                        child,
                        depth=depth + 1,
                        max_depth=max_depth,
                        max_items=max_items,
                        item_counter=item_counter,
                        active_ids=active_ids,
                    )
                return normalized_mapping
            return [
                _normalize_json_value(
                    child,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                    item_counter=item_counter,
                    active_ids=active_ids,
                )
                for child in value
            ]
        finally:
            active_ids.remove(identity)
    raise InvalidJobResultError()
