"""Best-effort token accounting for LangChain/OpenAI responses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


# Search a LangChain response for token counts. Different response types store
# token information in different places, so this function checks each format.
def collect_token_usage(value: Any) -> dict[str, int]:
    # Start at zero because some providers do not return token information.
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # The same message object can appear more than once inside a response. Keep
    # track of objects already counted so their tokens are not added twice.
    seen: set[int] = set()

    def visit(item: Any) -> None:
        # Stop when there is nothing to inspect or this object was already seen.
        if item is None or id(item) in seen:
            return
        seen.add(id(item))

        # Newer LangChain messages usually store counts in usage_metadata.
        usage = getattr(item, "usage_metadata", None)
        if isinstance(usage, Mapping):
            for key in totals:
                totals[key] += int(usage.get(key, 0) or 0)
        # Some providers use response_metadata instead. Only check it when
        # usage_metadata was not available, preventing double counting.
        response = getattr(item, "response_metadata", None)
        if not isinstance(usage, Mapping) and isinstance(response, Mapping):
            token_usage = response.get("token_usage") or response.get("usage")
            if isinstance(token_usage, Mapping):
                # OpenAI and LangChain sometimes use different names for the
                # same input/output token values.
                aliases = {
                    "input_tokens": ("input_tokens", "prompt_tokens"),
                    "output_tokens": ("output_tokens", "completion_tokens"),
                    "total_tokens": ("total_tokens",),
                }
                for target, keys in aliases.items():
                    totals[target] += int(next((token_usage[k] for k in keys if k in token_usage), 0) or 0)
        # Responses can contain nested dictionaries or lists of messages. Visit
        # each child so token counts from every model call are included.
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    # Begin the recursive search at the full response object.
    visit(value)

    # Calculate a total if the provider returned only input and output counts.
    if totals["total_tokens"] == 0:
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    return totals
