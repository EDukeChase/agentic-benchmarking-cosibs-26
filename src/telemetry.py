"""Best-effort token accounting for LangChain/OpenAI responses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def collect_token_usage(value: Any) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    seen: set[int] = set()

    def visit(item: Any) -> None:
        if item is None or id(item) in seen:
            return
        seen.add(id(item))
        usage = getattr(item, "usage_metadata", None)
        if isinstance(usage, Mapping):
            for key in totals:
                totals[key] += int(usage.get(key, 0) or 0)
        response = getattr(item, "response_metadata", None)
        if not isinstance(usage, Mapping) and isinstance(response, Mapping):
            token_usage = response.get("token_usage") or response.get("usage")
            if isinstance(token_usage, Mapping):
                aliases = {
                    "input_tokens": ("input_tokens", "prompt_tokens"),
                    "output_tokens": ("output_tokens", "completion_tokens"),
                    "total_tokens": ("total_tokens",),
                }
                for target, keys in aliases.items():
                    totals[target] += int(next((token_usage[k] for k in keys if k in token_usage), 0) or 0)
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    if totals["total_tokens"] == 0:
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    return totals
