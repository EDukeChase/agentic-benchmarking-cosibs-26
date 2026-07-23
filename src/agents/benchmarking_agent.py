from .authentication import token_provider
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage
from langchain.tools import tool
from src.core.schemas import LiteratureReviewResult
import json
import subprocess
import os
import shutil
from pathlib import Path
from src.settings.config import BenchmarkTaskConfig, LLMConfig
from src.settings.prompts import BENCHMARKING_SYSTEM_PROMPT

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



import re as _re

_MODEL_FAMILY_PATTERNS = [
    ("logisticregression", "logistic_regression"),
    ("linearregression", "linear_regression"),
    ("randomforest", "random_forest"),
    ("extratrees", "extra_trees"),
    ("xgboost", "xgboost"),
    ("histgradientboosting", "gradient_boosting"),
    ("gradientboosting", "gradient_boosting"),
    ("lightgbm", "lightgbm"),
    ("catboost", "catboost"),
    ("adaboost", "adaboost"),
    ("decisiontree", "decision_tree"),
    ("supportvectormachine", "svm"),
    ("svc", "svm"),
    ("svm", "svm"),
    ("kneighbors", "knn"),
    ("knearestneighbor", "knn"),
    ("knn", "knn"),
    ("naivebayes", "naive_bayes"),
    ("ridge", "ridge_regression"),
    ("lasso", "lasso_regression"),
    ("elasticnet", "elastic_net"),
]


def _normalize_model_family(name: str) -> str:
    """Map a folder/model name to a canonical family, tolerant of naming
    variants (e.g. 'random_forest' and 'random_forest_classifier' both ->
    'random_forest'). Falls back to a cleaned version of the raw name.
    """
    blob = _re.sub(r"[^a-z0-9]", "", name.lower())
    for pattern, canonical in _MODEL_FAMILY_PATTERNS:
        if pattern in blob:
            return canonical
    return blob or "unknown"

@tool
def execute_python(code: str, timeout: int = 480) -> str:
    """Execute Python code in the real project environment and return stdout/stderr."""
    timeout = min(max(int(timeout), 1), 900)  # raised from 480 -- see patch docstring
    try:
        result = subprocess.run(
            ["python", "-c", code],
            cwd="/app",
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return (
            f"Execution exceeded the {timeout}-second tool limit. "
            f"Partial stdout:\n{exc.stdout or ''}\n\n"
            f"Partial stderr:\n{exc.stderr or ''}"
        )
    return f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n\nExit code: {result.returncode}"

def build_benchmarking_agent(llm_config: LLMConfig = LLMConfig()):
    llm = ChatOpenAI(
        model=llm_config.model,
        temperature=llm_config.temperature,
        base_url="https://bpsmar-ai-openai-1.openai.azure.com/openai/v1/",
        api_key=token_provider,
        timeout=llm_config.timeout,
        max_retries=llm_config.max_retries,
    )
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

    # Collapse near-duplicate model folders (e.g. "random_forest" and
    # "random_forest_classifier" both existing for the same requested
    # model) down to one per family BEFORE the agent starts, so it never
    # wastes its step budget sanity-checking or testing folders it
    # shouldn't have needed to consider. Genuinely unrequested extras
    # (a different family entirely) are left untouched here.
    real_models_path_for_dedup = Path(f"/app{models_path}")
    expected_families_for_dedup = {
        _normalize_model_family(c.model_name) for c in literature_result.candidates
    }
    kept_by_family = {}
    duplicate_dirs = []
    if real_models_path_for_dedup.exists():
        for model_dir in sorted(real_models_path_for_dedup.iterdir()):
            if not model_dir.is_dir() or model_dir.name == "tmp":
                continue
            if not (model_dir / "model.py").exists():
                continue
            family = _normalize_model_family(model_dir.name)
            if family not in expected_families_for_dedup:
                continue
            if family in kept_by_family:
                duplicate_dirs.append(model_dir)
            else:
                kept_by_family[family] = model_dir
    if duplicate_dirs:
        print(
            f"NOTE: removing {len(duplicate_dirs)} duplicate model folder(s) "
            f"before benchmarking (keeping one per matched family): "
            f"{[d.name for d in duplicate_dirs]}"
        )
        for dup_dir in duplicate_dirs:
            shutil.rmtree(dup_dir)

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

    model_names = ", ".join(
        candidate.model_name for candidate in literature_result.candidates
    )
    human_message = f"""
    There are {len(literature_result.candidates)} models already implemented under
    {models_path}: {model_names}.

    {task_context}

    Inspect only the bounded paths permitted by the system prompt. Write the compact
    runner to /app{models_path}/run_benchmark.py and execute that exact file with
    execute_python. Write results to the real path /app{results_path}. Use at most
    one repair execution. Do not recursively grep or glob /app, /data,
    /generated_code, or patient_data_all.
    """
    # run agent with system and human messages
    response = _invoke_with_backoff(
        lambda: agent.invoke(
            {"messages": [SystemMessage(system_prompt), HumanMessage(human_message)]},
            config={"recursion_limit": 150},
        )
    )

    # verify that the agent wrote the results file to the real filesystem
    real_results_path = f"/app{results_path}"
    if not os.path.exists(real_results_path):
        raise RuntimeError(f"Agent never wrote {results_path} to the real filesystem.")

    context_path = Path(f"/app{models_path}/benchmark_context.json")
    if not context_path.exists():
        raise RuntimeError(
            f"Agent never wrote {models_path}/benchmark_context.json."
        )
    context = json.loads(context_path.read_text())
    required_context = {
        "training_prevalence",
        "prevalence_baseline_brier",
        "train_size",
        "validation_size",
        "test_size",
    }
    missing_context = required_context - context.keys()
    if missing_context:
        raise RuntimeError(
            "benchmark_context.json is missing required keys: "
            f"{sorted(missing_context)}"
        )

    split_count_keys = {
        "train_class_counts",
        "validation_class_counts",
        "test_class_counts",
    }
    if "class_counts" not in context and not split_count_keys.issubset(context):
        raise RuntimeError(
            "benchmark_context.json must contain either class_counts or all of: "
            f"{sorted(split_count_keys)}"
        )

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

