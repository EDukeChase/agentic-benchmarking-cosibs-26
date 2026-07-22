from .authentication import token_provider
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage
from langchain.tools import tool
from langchain_tavily import TavilySearch
from src.schemas import LiteratureReviewResult
import subprocess
import os
from pathlib import Path
from src.config import BenchmarkTaskConfig, LLMConfig
from src.prompts import BENCHMARKING_SYSTEM_PROMPT
# from uncertainty.uncertainty_quantification import calculate_uncertainty

@tool
def execute_python(code: str, timeout: int = 600) -> str:
    """Execute Python code in the real project environment and return stdout/stderr."""
    result = subprocess.run(
        ["python", "-c", code], cwd="/app", capture_output=True, text=True, timeout=timeout,
    )
    return f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n\nExit code: {result.returncode}"

def build_benchmarking_agent(max_search_results: int = 10, llm_config: LLMConfig = LLMConfig()):
    llm = ChatOpenAI(
        model=llm_config.model,
        temperature=llm_config.temperature,
        base_url="https://bpsmar-ai-openai-1.openai.azure.com/openai/v1/",
        api_key=token_provider,
        timeout=llm_config.timeout,
        max_retries=llm_config.max_retries,
    )
    # search_tool = TavilySearch(
    #     max_results=max_search_results,
    #     topic="general",
    # )
    return create_deep_agent(
        model=llm,
        tools=[execute_python],
        backend=FilesystemBackend(root_dir="/app", virtual_mode=False),
    )

def run_benchmarking_agent(
    agent,
    run_id: str,
    literature_result: LiteratureReviewResult,
    system_prompt_template: str = BENCHMARKING_SYSTEM_PROMPT,
    benchmark_task: BenchmarkTaskConfig | None = None,
):
    # the path for this specific run
    models_path = f"/generated_code/{run_id}"
    # the path where the agent should write the benchmarking results
    results_path = f"/generated_code/{run_id}/benchmark_results.json"

    # Use the configured prompt
    system_prompt = system_prompt_template.format(
        models_path=models_path,
        results_path=results_path,
    )

    task_context = (
        "Use the repository's benchmark task configuration."
        if benchmark_task is None
        else f"Use this resolved benchmark task configuration exactly: {benchmark_task!r}"
    )

    human_message = f"""
    There are {len(literature_result.candidates)} models already implemented under
    {models_path}, based on this literature review:
    {literature_result.model_dump_json(indent=2)}

    {task_context}

    Inspect only the bounded paths permitted by the system prompt, then write and run
    benchmarking test code for each model as described above. Use execute_python until
    it succeeds, and write results to {results_path} as instructed. Do not recursively
    grep or glob /app, /data, /generated_code, or patient_data_all.
    """
    # run agent with system and human messages
    response = agent.invoke({"messages": [SystemMessage(system_prompt), HumanMessage(human_message)]})

    # verify that the agent wrote the results file to the real filesystem
    real_results_path = f"/app{results_path}"
    if not os.path.exists(real_results_path):
        raise RuntimeError(f"Agent never wrote {results_path} to the real filesystem.")

    # the prompt requires a test_<model_name>_benchmark.py stub per model; the
    # reporting stage fails on any model missing one, so catch it here instead
    # of letting a mostly-successful run blow up two stages later.
    real_models_path = Path(f"/app{models_path}")
    missing = []
    for model_dir in sorted(real_models_path.iterdir()):
        if not model_dir.is_dir() or not (model_dir / "model.py").exists():
            continue
        if not (model_dir / f"test_{model_dir.name}_benchmark.py").exists():
            missing.append(model_dir.name)

    if missing:
        raise RuntimeError(
            f"Agent never wrote test_<model_name>_benchmark.py for: {missing}"
        )

    # predictions.json must be a flat array using the exact field names required
    # by the prompt (model, patient_id, true_diagnosis, probability, threshold,
    # generated_diagnosis). Each run's evaluator is LLM-generated and free to
    # invent different names, which silently breaks downstream tooling (the
    # reporting stage, plot_predictions.py) — catch drift here instead.
    real_predictions_path = Path(f"/app{models_path}/predictions.json")
    if not real_predictions_path.exists():
        raise RuntimeError(f"Agent never wrote {models_path}/predictions.json to the real filesystem.")

    predictions = json.loads(real_predictions_path.read_text())
    if not isinstance(predictions, list):
        raise RuntimeError(
            f"predictions.json must be a JSON array of per-patient records, got {type(predictions).__name__}."
        )

    required_keys = {"model", "patient_id", "true_diagnosis", "probability", "threshold", "generated_diagnosis"}
    missing_by_model: dict[str, set[str]] = {}
    for record in predictions:
        missing_keys = required_keys - record.keys()
        if missing_keys:
            model_name = record.get("model", "<unknown>")
            missing_by_model.setdefault(model_name, set()).update(missing_keys)

    if missing_by_model:
        details = ", ".join(f"{model}: {sorted(keys)}" for model, keys in missing_by_model.items())
        raise RuntimeError(f"predictions.json records missing required keys: {details}")

    return response

# Redo when uncertainty quantification is finished
def run_benchmarking_agent_with_uncertainty(
    agent,
    run_id: str,
    literature_result: LiteratureReviewResult,
    benchmark_task: BenchmarkTaskConfig | None = None,
    system_prompt_template: str = BENCHMARKING_SYSTEM_PROMPT,
    n_runs: int = 5,
):
    """Repeat the LLM benchmark and quantify variation between its responses.

    Every repetition receives the same models, resolved task, and prompt. The LLM
    is still free to make different implementation and reasoning choices, which is
    the variation this function is intended to measure. The result artifact is
    removed before each repetition so an old successful file cannot hide a failed
    attempt. Each newly observed result is saved in the uncertainty report before
    the next repetition overwrites the standard benchmark-results path.
    """
    import json
    from pathlib import Path

    from src.uncertainty.uncertainty_quantification import calculate_uncertainty

    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")

    responses: list[str] = []
    benchmark_results: list[dict] = []
    final_response = None
    results_path = Path(f"/app/generated_code/{run_id}/benchmark_results.json")

    for _ in range(n_runs):
        # Force run_benchmarking_agent to verify output created by this specific
        # repetition instead of accepting a file left by an earlier repetition.
        if results_path.exists():
            results_path.unlink()

        response = run_benchmarking_agent(
            agent=agent,
            run_id=run_id,
            literature_result=literature_result,
            benchmark_task=benchmark_task,
            system_prompt_template=system_prompt_template,
        )

        responses.append(str(response["messages"][-1].content))
        benchmark_results.append(json.loads(results_path.read_text()))
        final_response = response

    uncertainty = calculate_uncertainty(responses)
    uncertainty_path = Path(
        f"/app/generated_code/{run_id}/benchmark_uncertainty.json"
    )

    with uncertainty_path.open("w") as f:
        json.dump(
            {
                "agent_stage": "benchmarking",
                "uncertainty": uncertainty,
                "n_runs": n_runs,
                "benchmark_results_by_run": benchmark_results,
            },
            f,
            indent=2,
        )

    return {
        "response": final_response,
        "uncertainty": uncertainty,
    }

