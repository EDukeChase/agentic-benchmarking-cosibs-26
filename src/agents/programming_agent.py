from .authentication import token_provider
from deepagents import create_deep_agent
from pathlib import Path
from deepagents.backends import FilesystemBackend
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage
from src.core.schemas import LiteratureReviewResult, ModelCode
from pathlib import Path
import re
from src.settings.config import LLMConfig
from src.settings.prompts import PROGRAMMING_SYSTEM_PROMPT

from openai import RateLimitError as _RateLimitError
import time as _time

def _invoke_with_backoff(callable_fn, max_attempts=6, base_delay=20):
    """Retry callable_fn() on RateLimitError with linear backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            return callable_fn()
        except _RateLimitError:
            if attempt == max_attempts:
                raise
            delay = base_delay * attempt
            print(f"Rate limited (attempt {attempt}/{max_attempts}); waiting {delay}s before retrying...")
            _time.sleep(delay)

# from uncertainty.uncertainty_quantification import calculate_uncertainty

def _slugify(name: str) -> str:
    """Turns a model name into a safe, consistent folder name: lowercase, underscores only."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

def build_programming_agent(root_dir: str, max_search_results: int = 10, llm_config: LLMConfig = LLMConfig()):
    llm = ChatOpenAI(
      model = llm_config.model,
      temperature = llm_config.temperature,
      base_url = "https://bpsmar-ai-openai-1.openai.azure.com/openai/v1/",
      api_key = token_provider,
      timeout = llm_config.timeout,
      max_retries = llm_config.max_retries,
   )
    return create_deep_agent(
        model=llm,
        tools=[],
        backend=FilesystemBackend(root_dir=root_dir, virtual_mode=True),
    )

def run_programming_agent(agent, literature_result: LiteratureReviewResult, system_prompt_template: str = PROGRAMMING_SYSTEM_PROMPT):
    # build the exact folder-name list
    name_mapping = "\n".join(
        f"- {c.model_name}  ->  folder name: {_slugify(c.model_name)}"
        for c in literature_result.candidates
    )

    # Use the centrally configured prompt. Custom templates must accept name_mapping.
    system_prompt = system_prompt_template.format(name_mapping=name_mapping)

    # the literature review JSON is passed as a human message so the agent can read it
    literature_json = literature_result.model_dump_json(indent=2)
    human_message = f"""
    Implement the {len(literature_result.candidates)} models described in this literature
    review:
    {literature_json}

    Search the web if additional documentation and implementation details are needed.
    Execute sanity checks using the Python tool before finishing, and follow the folder
    naming rule exactly.
    """

    return _invoke_with_backoff(
        lambda: agent.invoke({"messages": [SystemMessage(system_prompt), HumanMessage(human_message)]})
    )

def collect_generated_models(output_dir: str) -> list[ModelCode]:
    # find path with generated models
    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(
            f"Programming output directory was not created: {output_path}"
        )

    # get code and documentation file for each model
    models = []
    for model_dir in output_path.iterdir():
        if not model_dir.is_dir():
            continue
        code_file = model_dir / "model.py"
        docs_file = model_dir / "docs.md"
        if not code_file.exists():
            print(f"Warning: skipping '{model_dir.name}' — no model.py found")
            continue

        # add the code and documentation to a ModelCode object
        models.append(ModelCode(
            model_name=model_dir.name,
            code=code_file.read_text(),
            documentation=docs_file.read_text() if docs_file.exists() else "No documentation provided.",
        ))
    return models

# Finding uncertainty
"""
def run_programming_agent_with_uncertainty(
    agent,
    literature_result,
    output_dir: str,
    n_runs=5,
):
    all_code_outputs = []

    for i in range(n_runs):

        run_programming_agent(agent, literature_result)

        models = collect_generated_models(output_dir)

        text = "\n".join(
            model.code
            for model in models
        )

        all_code_outputs.append(text)

    uncertainty = calculate_uncertainty(all_code_outputs)

    return {
    "models": models,
    "uncertainty": uncertainty,
    }
    """
