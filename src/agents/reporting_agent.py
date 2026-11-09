"""The reporting stage: one model call that writes report.md from the earlier stages' output."""

from __future__ import annotations

import json
from pathlib import Path

from langchain.messages import HumanMessage, SystemMessage

from src.core.schemas import LiteratureReviewResult
from src.settings.config import LLMConfig
from src.settings.prompts import REPORTING_STAGE_INSTRUCTIONS
from src.settings.study_prompt import StudyPrompt, compose

from .chat_model import build_chat_model


def build_reporting_agent(llm_config: LLMConfig = LLMConfig()):
    return build_chat_model(llm_config)


def write_report(
    llm,
    study_prompt: StudyPrompt,
    out_dir: str | Path,
    literature_result: LiteratureReviewResult,
    benchmarking_message: str,
):
    """Ask the model for report.md and return the raw model response.

    The caller writes the text to {out_dir}/report.md. Missing or unreadable
    results.json is passed on as such, so the report states it instead of
    inventing numbers.
    """
    results_path = Path(out_dir) / "results.json"
    try:
        results_text = results_path.read_text()
    except OSError:
        results_text = "results.json was not written."

    request = (
        "results.json written by the benchmarking stage:\n"
        f"{results_text}\n\n"
        "Final message of the benchmarking stage:\n"
        f"{benchmarking_message}\n\n"
        "Literature review that chose methods 3 to 5:\n"
        f"{literature_result.model_dump_json(indent=2, exclude={'grounding'})}"
    )
    return llm.invoke([
        SystemMessage(compose(study_prompt, REPORTING_STAGE_INSTRUCTIONS)),
        HumanMessage(request),
    ])
