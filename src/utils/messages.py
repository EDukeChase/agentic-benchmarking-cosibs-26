"""Text of a chat model reply.

With the Responses API a reply's content is a list of blocks (reasoning, text), not a
string, so str(message.content) would give the list's repr. This returns the text only.
"""

from __future__ import annotations


def message_text(message) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):  # langchain-core 1.x: a property that is a str
        return str(text)
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in (None, "text", "output_text"):
            parts.append(str(block.get("text", "")))
    return "".join(parts)
