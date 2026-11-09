"""The fixed study prompt, given word for word to every agent.

The prompt file is rendered outside the repository (the repository is readable
by the agents, the study plan is not) and handed to the pipeline by path. Its
SHA-256 is recorded for every run, so each run can be checked against the text
the other architecture received.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StudyPrompt:
    text: str
    sha256: str


def load_study_prompt(path: str | Path, expected_sha256: str | None = None) -> StudyPrompt:
    """Read the rendered study prompt. The bytes are hashed exactly as stored."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError(
            f"Study prompt {path} has SHA-256 {digest}, expected {expected_sha256}."
        )
    text = raw.decode("utf-8")
    if "{data_dir}" in text or "{out_dir}" in text:
        raise ValueError("The study prompt still has {data_dir} or {out_dir} placeholders; use the rendered copy.")
    return StudyPrompt(text=text, sha256=digest)


def compose(study_prompt: StudyPrompt, stage_instructions: str) -> str:
    """The study prompt unchanged, then what is special about this stage."""
    return (
        "The following task description is the same for every stage of the pipeline.\n\n"
        "<study_prompt>\n"
        f"{study_prompt.text.rstrip()}\n"
        "</study_prompt>\n\n"
        f"{stage_instructions.strip()}\n"
    )
