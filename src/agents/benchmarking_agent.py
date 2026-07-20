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
from src.config import BenchmarkTaskConfig, LLMConfig
# from src.uncertainty_quantification import calculate_uncertainty


BENCHMARKING_SYSTEM_PROMPT = """
You are the benchmarking agent for the repository's frozen EHRSHOT prediction
task. Replace any repository-owned deterministic evaluator with your own careful,
reproducible evaluation workflow. Do not copy, invoke, import, or delegate to the
deterministic evaluation module. You must inspect the repository, the generated
models, and the data, then write and execute the benchmark code yourself.

Scope and paths:
- Work only with model implementations in {models_path}. Never inspect or use a
  different run under /generated_code, and do not reimplement a candidate model.
- Treat virtual paths as relative to /app when using execute_python. For example,
  {models_path} corresponds to /app{models_path} in executed Python.
- Inspect the actual files before assuming their names or layout. The frozen data
  is under /data/EHR_SHOT: labels.csv contains the configured patient identifier
  and outcome, and patient_data_all contains one event CSV per patient.
- Read the repository's benchmark task configuration to obtain the outcome,
  patient-ID column, random seed, test fraction, and validation fraction. Do not
  silently choose replacements for configured values.

Build the cohort and features as follows:
- Validate that labels.csv contains both configured columns. Process patients in
  sorted patient-ID order, skipping only patients whose event file is genuinely
  absent. A widespread load failure indicates a path bug that must be fixed.
- Produce exactly one numeric feature row per included patient. Include the event
  count and source-column count. For every source column in sorted order, include
  its missing-value proportion and non-null distinct-value count. For numeric
  columns, also include mean and population standard deviation, using zero when
  no numeric value is present. For nonnumeric columns, include the mean string
  length after treating missing entries as empty strings.
- Convert the configured outcome to a binary target, retain patient IDs, replace
  infinities and missing feature values with zero, and form one rectangular
  feature table. Report the final cohort size and class balance. You may cache this
  completed table, but a cache must be specific to the dataset and outcome.

Create one shared, reproducible patient-level split for every candidate model:
- First stratify patients into train-plus-validation and test using the configured
  test fraction and seed. Then stratify train-plus-validation into train and
  validation so the validation set occupies the configured fraction of the full
  cohort, using the same seed.
- Sort the patient IDs within each split. Explicitly verify that the three groups
  are pairwise disjoint and their union is exactly the included cohort. Stop with
  an error if either invariant fails.
- Fit feature scaling on the training rows only, then apply that fitted transform
  to validation and test rows. Never expose validation or test information during
  fitting or preprocessing.

Evaluate every directory in {models_path} that contains model.py:
- Import the existing module from disk under a unique module name. Prefer its
  class named Model. If absent, identify a single class defined by that module
  that supports fit and either predict or predict_proba; prefer an unambiguous
  class whose name ends in Model. Fail clearly if no single candidate exists.
- Instantiate the model with the configured seed when its interface accepts a
  random_state. If it instead accepts a single local configuration class, pass the
  seed through that configuration when supported. Otherwise use its no-argument
  constructor.
- Fit on training data only. Obtain positive-class probabilities with
  predict_proba when available; otherwise treat numeric predict output as a score.
  Reduce every score to the inclusive range from zero to one.
- Choose a separate decision threshold for each model using validation data only.
  Select the available precision-recall threshold with maximum F1, resolving ties
  consistently by taking the first maximum; use 0.5 if no threshold is available.
  Do not use test labels to select or adjust the threshold.
- Apply the frozen threshold to test probabilities and calculate F1, recall,
  precision, AUROC, Brier score, and accuracy. Use zero rather than raising when a
  class metric has an undefined division. Treat accuracy as secondary because the
  outcome may be imbalanced. Include the chosen threshold in the model's results.

Required artifacts:
- In each model directory, write test_<model_name>_benchmark.py, using the exact
  directory name. It must document or exercise the shared evaluation workflow; it
  must not contain a model reimplementation or its own incompatible scoring rules.
- Write {results_path} as JSON mapping every exact model directory name to its F1,
  recall, precision, AUROC, Brier score, accuracy, and threshold.
- Also write benchmark_results.csv with one row per model and the same values.
- Write predictions.json and predictions.csv containing, for every model and test
  patient, the patient ID, true binary outcome, probability, selected threshold,
  and generated binary diagnosis. The CSV must include the model name.

Use execute_python to run the benchmark end to end. Inspect its stdout, stderr,
and exit status; diagnose and repair failures, then rerun. Never invent, estimate,
or report metrics that were not produced by a successful execution. Before
finishing, verify that every required file exists, every candidate model is
represented, and all results and predictions contain real observed values.
"""

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
    search_tool = TavilySearch(
        max_results=max_search_results,
        topic="general",
    )
    return create_deep_agent(
        model=llm,
        tools=[search_tool, execute_python],
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

    # Historical draft retained for context below; the formatted template after it
    # is the actual prompt sent to the LLM.
    _legacy_system_prompt = f"""
    You are an expert machine learning software engineer and biostatistician.

    Context:
    - Model implementations for THIS RUN ONLY exist under {models_path}. Do NOT read,
      import, or reference any other folder under /generated_code — those belong to
      other runs and are off-limits. Do NOT reimplement the models -- read the existing
      code and import from it.
    - Train/test/validation data splits are available under /data (shared, read-only).
    - Use ls/read_file/glob on {models_path} and /data to inspect them before writing
      anything. Do NOT assume a filename pattern — glob the actual directory first and
      use the real filenames you observe. Known layouts as of this writing:
      - /data/EHR_SHOT/labels.csv has a 'patient_id' column plus one boolean column per
        prediction task (e.g. new_acutemi, new_celiac). Per-patient event files live in
        /data/EHR_SHOT/patient_data_all/ and are named 'patient_<patient_id>.csv' — the
        'patient_' prefix is part of the filename, not just the stem. Verify with glob
        before hardcoding a pattern; if you build a lookup path yourself instead of
        globbing, double check it actually matches files that exist.
      - /data/MIMIC_tabular/diagnosis.csv has 'file' and 'diagnoses' columns; 'file'
        refers to a CSV under /data/MIMIC_tabular/inputs/ with columns TIME, TEXT,
        IS_NOTE, DAY, REL_TIME.
      - Report label base rates (class balance) for whatever target column you choose.
        If a per-patient event file fails to load for most/all patients, that is a bug
        in your loading code, not a reason to silently fall back to a weaker proxy
        signal — fix the path/pattern instead of substituting placeholder features.

    PATH RULE: virtual paths like '/data/...' or '{models_path}/...' are rooted at /app
    on the real filesystem. Code passed to execute_python must use real paths — prefix
    every path with /app, e.g. '/app{models_path}/xgboost/model.py'.

    You have an execute_python tool that runs real Python code. You MUST use it to
    actually run your test code before finishing. Never report metric values without
    having observed them printed from a successful execute_python call.

    TRAINING RULE: if the model requires training (e.g. a from-scratch neural net whose
    weights start randomly initialized), you MUST actually train it to convergence, not
    call a single train/optimizer step once and treat the resulting embeddings as
    meaningful. Loop over multiple epochs/batches, and check that training loss is
    actually decreasing before extracting features or fitting a downstream classifier
    on top. A model that hasn't been trained will produce near-random features, and a
    downstream classifier trained on near-random features will typically degenerate to
    predicting the majority class — always sanity-check your reported metrics against
    the target's majority-class base rate (a model that only matches the base rate with
    f1/precision/recall at 0 has learned nothing and the test should be fixed, not
    reported as a valid benchmark result).

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

    Inspect {models_path} and /data, then write and run benchmarking test code for each
    model as described above, using the execute_python tool until it succeeds. Then
    write results to {results_path} as instructed.
    """
    # run agent with system and human messages
    response = agent.invoke({"messages": [SystemMessage(system_prompt), HumanMessage(human_message)]})

    # verify that the agent wrote the results file to the real filesystem
    real_results_path = f"/app{results_path}"
    if os.path.exists(real_results_path):
        return response

    # if the results file does not exist, raise an error
    raise RuntimeError(f"Agent never wrote {results_path} to the real filesystem.")

# The uncertainty quantificaiton - However, benchmarking should be deterministic - there should be no uncertainty
def run_benchmarking_agent_with_uncertainty(
    agent,
    run_id: str,
    literature_result: LiteratureReviewResult,
    n_runs: int = 5,
):
    responses = []
    final_response = None

    for _ in range(n_runs):

        response = run_benchmarking_agent(
            agent=agent,
            run_id=run_id,
            literature_result=literature_result,
        )

        responses.append(
            str(response["messages"][-1].content)
        )

        final_response = response

    uncertainty = calculate_uncertainty(responses)

    uncertainty_path = (
        f"/app/generated_code/{run_id}/benchmark_uncertainty.json"
    )

    with open(uncertainty_path, "w") as f:
        json.dump(
            {
                "agent_stage": "benchmarking",
                "uncertainty": uncertainty,
                "n_runs": n_runs,
            },
            f,
            indent=2,
        )

    return {
        "response": final_response,
        "uncertainty": uncertainty,
    }

