"""Shared name-normalization helpers.

Kept separate so the programming, reporting, and orchestration layers derive
folder/model identifiers from literature-review names the exact same way.
"""

from __future__ import annotations

import re


def slugify(name: str) -> str:
    """Turn a model name into a safe, consistent folder name: lowercase, underscores only."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
