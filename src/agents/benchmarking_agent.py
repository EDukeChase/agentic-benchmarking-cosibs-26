from .authentication import token_provider
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage, SystemMessage, HumanMessage
from langchain.tools import tool
from langchain_tavily import TavilySearch
from src.schemas import LiteratureReviewResult
import subprocess
import os

@tool
def execute_python(code: str, timeout: int = 600) -> str:
    """Execute Python code in the real project environment and return stdout/stderr."""
    result = subprocess.run(
        ["python", "-c", code], cwd="/app", capture_output=True, text=True, timeout=timeout,
    )
    return f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n\nExit code: {result.returncode}"

def build_benchmarking_agent(max_search_results: int = 10):
    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        base_url="https://bpsmar-ai-openai-1.openai.azure.com/openai/v1/",
        api_key=token_provider,
        timeout=120,
        max_retries=2,
    )
    search_tool = TavilySearch(
        max_results=max_search_results,
        topic="general",
    )
    return create_deep_agent(
        model=llm,
        tools=[search_tool, execute_python],
        backend=FilesystemBackend(root_dir="/app", virtual_mode=False),
    )

def run_benchmarking_agent(agent, run_id: str, literature_result: LiteratureReviewResult) -> None:
    # the path for this specific run
    models_path = f"/generated_code/{run_id}"
    # the path where the agent should write the benchmarking results
    results_path = f"/generated_code/{run_id}/benchmark_results.json"

    system_prompt = f"""
    You are an expert machine learning software engineer and biostatistician.

    Context:
    - Model implementations for THIS RUN ONLY exist under {models_path}. Do NOT read,
      import, or reference any other folder under /generated_code — those belong to
      other runs and are off-limits. Do NOT reimplement the models -- read the existing
      code and import from it.
    - Train/test/validation data splits are available under /data (shared, read-only).
    - Use ls/read_file/glob on {models_path} and /data to inspect them before writing
      anything.

    PATH RULE: virtual paths like '/data/...' or '{models_path}/...' are rooted at /app
    on the real filesystem. Code passed to execute_python must use real paths — prefix
    every path with /app, e.g. '/app{models_path}/xgboost/model.py'.

    You have an execute_python tool that runs real Python code. You MUST use it to
    actually run your test code before finishing. Never report metric values without
    having observed them printed from a successful execute_python call.

    For each model found under {models_path}, write a test module named
    test_<model_name>_benchmark.py — using the EXACT model folder name as <model_name> —
    placed directly inside that same model's folder (e.g.
    {models_path}/xgboost/test_xgboost_benchmark.py). Each test module should load the
    data splits, fit/evaluate the model, and compute accuracy, F1, precision, recall,
    AUROC, and Brier score.

    MANDATORY FINAL STEP: write ALL results to {results_path} (a real disk path is
    /app{results_path}) as a JSON object mapping each model's exact folder name to a
    dict containing accuracy, f1, precision, recall, auroc, and brier for each model.
    Do not consider the task complete until this file exists with real observed numbers
    in it.
    """

    human_message = f"""
    There are {len(literature_result.candidates)} models already implemented under
    {models_path}, based on this literature review:
    {literature_result.model_dump_json(indent=2)}

    Inspect {models_path} and /data, then write and run benchmarking test code for each
    model as described above, using the execute_python tool until it succeeds. Then
    write results to {results_path} as instructed.
    """
    # run agent with system and human messages
    agent.invoke({"messages": [SystemMessage(system_prompt), HumanMessage(human_message)]})

    # verify that the agent wrote the results file to the real filesystem
    real_results_path = f"/app{results_path}"
    if os.path.exists(real_results_path):
        return

    # if the results file does not exist, raise an error
    raise RuntimeError(f"Agent never wrote {results_path} to the real filesystem.")
