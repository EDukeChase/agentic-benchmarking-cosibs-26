from .authentication import token_provider
from deepagents import create_deep_agent
from pathlib import Path
#from .literature_agent import literature_agent
from deepagents.backends import FilesystemBackend
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool
from langchain_tavily import TavilySearch
from src.schemas import LiteratureReviewResult, ModelCode
from pathlib import Path
import re

def _slugify(name: str) -> str:
    """Turns a model name into a safe, consistent folder name: lowercase, underscores only."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

def build_programming_agent(root_dir: str, max_search_results: int = 10):
    llm = ChatOpenAI(
      model = "gpt-5.4-mini",
      base_url = "https://bpsmar-ai-openai-1.openai.azure.com/openai/v1/",
      api_key = token_provider,
   )
    search_tool = TavilySearch(
      max_results = max_search_results,
      topic = "general",
   )
    return create_deep_agent(
        model=llm,
        tools=[search_tool],
        backend=FilesystemBackend(root_dir=root_dir, virtual_mode=True),
    )

def run_programming_agent(agent, literature_result: LiteratureReviewResult) -> None:
    # build the exact folder-name list
    name_mapping = "\n".join(
        f"- {c.model_name}  ->  folder name: {_slugify(c.model_name)}"
        for c in literature_result.candidates
    )

    system_prompt = f"""
    You are an expert machine learning software engineer and biostatistician.
    Your task is to implement the machine learning models specified in the literature
    review below. Write clean, modular, well-documented Python code suitable for
    benchmarking on EHRSHOT datasets. When documentation is incomplete, identify the
    missing assumptions explicitly instead of inventing behavior. Test your
    implementation when possible using the Python execution tool.

    FOLDER NAMING RULE (mandatory, no exceptions):
    You must create EXACTLY one folder per model, using EXACTLY the folder names below —
    do not invent your own names, do not add version suffixes, do not change casing:
    {name_mapping}

    Each folder must contain exactly two files:
    - model.py — the complete implementation (architecture, training, evaluation combined
      into one importable module)
    - docs.md — a short markdown file documenting implementation decisions, assumptions
      made where source documentation was incomplete, and known limitations

    You are working in a fresh, empty working directory ("/") — this is expected, not an
    error. Create all folders and files directly at the root of your filesystem.
    """

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

    agent.invoke({"messages": [SystemMessage(system_prompt), HumanMessage(human_message)]})

def collect_generated_models(output_dir: str) -> list[ModelCode]:
    # find path with generated models
    output_path = Path(output_dir)

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
            model_name=model_dir.name,   # matches the slugified name exactly, by construction
            code=code_file.read_text(),
            documentation=docs_file.read_text() if docs_file.exists() else "No documentation provided.",
        ))
    return models
