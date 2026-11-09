"""Literature discovery with Gemini on Vertex AI and Google Search grounding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from google import genai
from google.genai import types
from pydantic import BaseModel, conlist, create_model

from src.core.schemas import GeneratedModel, LiteratureReviewResult
from src.settings.config import GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_PROJECT, LLMConfig
from src.settings.prompts import LITERATURE_STAGE_INSTRUCTIONS
from src.settings.study_prompt import StudyPrompt, compose


@dataclass(frozen=True)
class LiteratureAgent:
    """Vertex AI client and generation settings for literature discovery."""

    client: genai.Client
    model: str
    temperature: float
    # Vertex calls this thinking_level; None leaves the model's default.
    reasoning_effort: str | None = None


def build_literature_agent(llm_config: LLMConfig) -> LiteratureAgent:
    """Create a Gemini client that authenticates through Google Cloud ADC."""

    client = genai.Client(
        vertexai=True,
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
        http_options=types.HttpOptions(api_version="v1"),
    )
    return LiteratureAgent(
        client=client,
        model=llm_config.model,
        temperature=llm_config.temperature,
        reasoning_effort=llm_config.reasoning_effort,
    )


def _bounded_response_model(num_models: int) -> type[BaseModel]:
    """Build a response schema containing exactly ``num_models`` candidates."""

    candidates_type = conlist(
        GeneratedModel,
        min_length=num_models,
        max_length=num_models,
    )
    return create_model(
        f"LiteratureReviewWith{num_models}Candidates",
        candidates=(candidates_type, ...),
    )


def _validate_result(result: LiteratureReviewResult, num_models: int) -> None:
    if len(result.candidates) != num_models:
        raise RuntimeError(
            f"Gemini returned {len(result.candidates)} candidate(s); "
            f"expected exactly {num_models}."
        )

    for candidate in result.candidates:
        parsed = urlparse(candidate.resource_link)
        if parsed.scheme != "https" or not parsed.netloc:
            raise RuntimeError(
                f"{candidate.model_name!r} returned an invalid source URL: "
                f"{candidate.resource_link!r}"
            )


def extract_grounding(response: Any) -> dict[str, Any] | None:
    """What the search tool reported for one response: the queries and the pages.

    Best effort. The field names follow google-genai's GroundingMetadata; if the
    response has none (or has a different shape), this returns None instead of
    failing the stage, because the metadata is a record and the stage does not
    depend on it.
    """
    try:
        candidates = getattr(response, "candidates", None) or []
        metadata = getattr(candidates[0], "grounding_metadata", None) if candidates else None
        if metadata is None:
            return None
        sources = []
        for chunk in getattr(metadata, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web is not None:
                sources.append({"uri": getattr(web, "uri", None), "title": getattr(web, "title", None)})
        return {
            "web_search_queries": list(getattr(metadata, "web_search_queries", None) or []),
            "sources": sources,
        }
    except Exception:  # noqa: BLE001 - a record must never break the stage
        return None


def run_literature_review(
    agent: LiteratureAgent,
    num_models: int,
    study_prompt: StudyPrompt,
) -> LiteratureReviewResult:
    """Search the public web and return implementation-ready model candidates."""

    if num_models < 1:
        raise ValueError("num_models must be at least 1")

    prompt = compose(study_prompt, LITERATURE_STAGE_INSTRUCTIONS.format(num_models=num_models))

    # Vertex AI does not support controlled JSON-schema generation and Google
    # Search grounding together for every Gemini model. Search first, then use a
    # second non-search request to normalize the grounded findings.
    search_response = agent.client.models.generate_content(
        model=agent.model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=agent.temperature,
            thinking_config=(
                types.ThinkingConfig(thinking_level=agent.reasoning_effort)
                if agent.reasoning_effort
                else None
            ),
        ),
    )

    if not search_response.text:
        raise RuntimeError("Gemini returned no grounded literature findings")
    grounding = extract_grounding(search_response)

    format_prompt = f"""
Convert the grounded research below into the required response schema.

Rules:
- Return exactly {num_models} candidates.
- Preserve only facts and URLs present in the grounded research.
- resource_link must be the direct HTTPS URL of the source named in the research:
  a code repository, or a well-documented paper or documentation page. Not a
  search page and not an invented URL.
- source_type is "repository" when resource_link is a code repository, and
  "paper_or_documentation" when it is a paper or a documentation page.
- Do not add a candidate whose source is not in the research.
- Keep implementation details in summary and selection reasoning in rationale.

Grounded research:
{search_response.text}
"""

    structured_response = agent.client.models.generate_content(
        model=agent.model,
        contents=format_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_bounded_response_model(num_models),
            temperature=agent.temperature,
        ),
    )

    if structured_response.parsed is not None:
        raw_result = structured_response.parsed
    elif structured_response.text:
        raw_result = json.loads(structured_response.text)
    else:
        raise RuntimeError("Gemini returned no structured literature-review content")

    if isinstance(raw_result, BaseModel):
        raw_result = raw_result.model_dump()

    result = LiteratureReviewResult.model_validate(raw_result)
    _validate_result(result, num_models)
    return result.model_copy(update={"grounding": grounding})
