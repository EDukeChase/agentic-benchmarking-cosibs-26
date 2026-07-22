from .authentication import token_provider
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from src.schemas import LiteratureReviewResult
from langchain_tavily import TavilySearch
from src.config import LLMConfig
from src.prompts import LITERATURE_SYSTEM_PROMPT
from uncertainty.uncertainty_quantification import calculate_uncertainty


SYSTEM_PROMPT = """
You are an expert scientist in the field of biostatistics who is working on a research project.
Your research group is tasked with benchmarking the performance of various new machine learning 
models to predict patient outcomes based on clinical data, specifically data in the format of 
EHRSHOT. Can you please provide a list of candidate models to benchmark, along with a summary 
of the documentation for each model?
"""


def build_literature_agent(max_search_results: int = 10, llm_config: LLMConfig = LLMConfig()):
    llm = ChatOpenAI(
        model=llm_config.model,
        temperature=llm_config.temperature,
        base_url="https://bpsmar-ai-openai-1.openai.azure.com/openai/v1/",
        api_key=token_provider,
        timeout=llm_config.timeout,
        max_retries=llm_config.max_retries,
    )
    search_tool = TavilySearch(max_results=max_search_results, topic="general")
    agent = create_agent(
        model=llm,
        tools=[search_tool],
        response_format=ToolStrategy(LiteratureReviewResult),   # <-- the key line
    )
    return agent


from __future__ import annotations

import json
import os
from typing import Any

from langchain.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.schemas import LiteratureReviewResult
from src.config import LLMConfig
from src.prompts import LITERATURE_SYSTEM_PROMPT
from uncertainty.uncertainty_quantification import calculate_uncertainty


GOOGLE_CLOUD_PROJECT = os.environ.get(
    "GOOGLE_CLOUD_PROJECT",
    "gac-som-dbmi-bpsmar-app-59",
)

GOOGLE_CLOUD_LOCATION = os.environ.get(
    "GOOGLE_CLOUD_LOCATION",
    "global",
)

# def run_literature_review(agent, num_models: int, system_prompt: str = LITERATURE_SYSTEM_PROMPT) -> LiteratureReviewResult:
#     messages = [
#         SystemMessage(system_prompt),
#         HumanMessage(
#             f"Please look online for {num_models} candidate models to benchmark, and "
#             "provide a summary of the documentation for each. Make sure there is enough "
#             "information for the next scientist to implement the model using just your "
#             "information."
#         ),
#     ]
#     result = agent.invoke({"messages": messages})
#     return result["structured_response"]

def run_literature_review(
    agent,
    num_models: int,
    system_prompt: str = LITERATURE_SYSTEM_PROMPT,
) -> LiteratureReviewResult:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"""
Search the public web for exactly {num_models} candidate models that could be
benchmarked for prediction of patient outcomes from EHRSHOT-style clinical data.

For every candidate:

- provide the exact model name
- identify an official implementation or reliable documentation source
- explain the model at enough technical depth for another scientist to implement it
- explain why it is appropriate for this benchmarking problem
- explain important limitations
- provide a valid source URL
- do not invent papers, documentation, package names, or URLs

Prefer, in this order:

1. Peer-reviewed methodological papers
2. Official package documentation
3. Official repository documentation
4. Reputable academic sources

Make sure the final response follows the requested JSON schema exactly.
"""
        ),
    ]

    response = agent.invoke(messages)

    # With response_mime_type="application/json", Gemini normally returns
    # the JSON as response.content.
    content: Any = response.content

    if isinstance(content, str):
        raw_result = json.loads(content) #takes the json string and converts it into a python object
    elif isinstance(content, dict):
        raw_result = content
    elif isinstance(content, list):
        # Some Gemini/LangChain versions return content blocks.
        text_blocks = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"#gets the text out of the list
        ]
        raw_text = "".join(text_blocks)#joins the text pieces into a single string

        if not raw_text:
            raise RuntimeError(
                f"Gemini returned content blocks but no text: {content!r}"
            )

        raw_result = json.loads(raw_text)
    else:
        raise RuntimeError(
            "Unexpected Gemini response content type: "
            f"{type(content).__name__}"
        )

    return LiteratureReviewResult.model_validate(raw_result)



# # [Returning Uncertainty Quantification below]

# def run_literature_review_with_uncertainty(
#     agent,
#     num_models: int,
#     n_runs: int = 5,
# ):
#     outputs = []

#     for _ in range(n_runs):
#         result = run_literature_review(agent, num_models)

#         text = "\n".join(
#             f"{c.model_name}: {c.summary}"
#             for c in result.candidates
#         )

#         outputs.append(text)

#     uncertainty = calculate_uncertainty(outputs)

#     return {
#         "result": result,
#         "uncertainty": uncertainty,
#         "all_outputs": outputs,
#     }

def run_literature_review_with_uncertainty(
    agent,
    num_models: int,
    n_runs: int = 5,
):
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1.")

    results: list[LiteratureReviewResult] = []
    outputs: list[str] = []

    for _ in range(n_runs):
        result = run_literature_review(agent, num_models)
        results.append(result)

        text = "\n".join(
            f"{candidate.model_name}: {candidate.summary}"
            for candidate in result.candidates
        )
        outputs.append(text)#builds one long text string from all the candidate models.

    uncertainty = calculate_uncertainty(outputs)

    return {
        # This is still the final run, matching your previous behavior.
        "result": results[-1],
        "uncertainty": uncertainty,
        "all_outputs": outputs,
        # Keeping every structured run is useful for auditing.
        "all_results": results,
    }

