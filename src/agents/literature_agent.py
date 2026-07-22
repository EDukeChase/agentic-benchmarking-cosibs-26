"""Literature discovery with Gemini on Vertex AI and Google Search grounding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urlparse

from google import genai
from google.genai import types
from pydantic import BaseModel, conlist, create_model

from src.core.schemas import GeneratedModel, LiteratureReviewResult
from src.settings.config import GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_PROJECT, LLMConfig
from src.settings.prompts import LITERATURE_SYSTEM_PROMPT


@dataclass(frozen=True)
class LiteratureAgent:
    """Vertex AI client and generation settings for literature discovery."""

    client: genai.Client
    model: str
    temperature: float


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
                f"{candidate.model_name!r} returned an invalid source-code URL: "
                f"{candidate.resource_link!r}"
            )


def run_literature_review(
    agent: LiteratureAgent,
    num_models: int,
    system_prompt: str = LITERATURE_SYSTEM_PROMPT,
) -> LiteratureReviewResult:
    """Search the public web and return implementation-ready model candidates."""

    if num_models < 1:
        raise ValueError("num_models must be at least 1")

    prompt = f"""
{system_prompt}

Use Google Search to find exactly {num_models} distinct candidate models for
predicting patient outcomes from longitudinal EHRSHOT-style clinical data.

For every candidate:
- Use a primary paper or authoritative technical description.
- Find an official, author-maintained, or otherwise trusted source-code repository.
- Set resource_name to the implementation or repository name.
- Set resource_link to the direct HTTPS URL of that source-code repository.
- Summarize the inputs, architecture, training objective, and implementation details
  needed by the programming agent.
- Explain why the model fits this benchmark and identify important limitations.

Do not invent papers, implementation details, repository names, or URLs. Exclude a
candidate if you cannot find usable source code. Gather evidence for exactly
{num_models} candidates and include the direct repository URLs in your findings.
"""

    # Vertex AI does not support controlled JSON-schema generation and Google
    # Search grounding together for every Gemini model. Search first, then use a
    # second non-search request to normalize the grounded findings.
    search_response = agent.client.models.generate_content(
        model=agent.model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=agent.temperature,
        ),
    )

    if not search_response.text:
        raise RuntimeError("Gemini returned no grounded literature findings")

    format_prompt = f"""
Convert the grounded research below into the required response schema.

Rules:
- Return exactly {num_models} candidates.
- Preserve only facts and URLs present in the grounded research.
- resource_link must be the direct HTTPS source-code repository URL, not a paper,
  search page, or invented URL.
- Do not add a candidate whose source-code repository is not in the research.
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
            temperature=0.0,
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
    return result


def run_literature_review_with_uncertainty(
    agent: LiteratureAgent,
    num_models: int,
    n_runs: int = 5,
):
    """Repeat discovery and quantify variation between returned summaries."""

    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")

    # Loading the sentence-transformer is expensive and may download model files,
    # so keep it out of normal literature-agent startup.
    from src.uncertainty.uncertainty_quantification import calculate_uncertainty

    results: list[LiteratureReviewResult] = []
    outputs: list[str] = []
    for _ in range(n_runs):
        result = run_literature_review(agent, num_models)
        results.append(result)
        outputs.append(
            "\n".join(
                f"{candidate.model_name}: {candidate.summary}"
                for candidate in result.candidates
            )
        )

    return {
        "result": results[-1],
        "uncertainty": calculate_uncertainty(outputs),
        "all_outputs": outputs,
        "all_results": results,
    }
