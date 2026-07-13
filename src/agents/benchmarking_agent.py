from authentication import token_provider
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from literature_agent import NUMBER_OF_MODELS
from literature_agent import response
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage
from langchain.tools import tool
from langchain_tavily import TavilySearch
import subprocess
import json
import os

MAX_SEARCH_RESULTS = 10

llm = ChatOpenAI(
    model="gpt-5.4-mini",
    base_url="https://bpsmar-ai-openai-1.openai.azure.com/openai/v1/",
    api_key=token_provider,
    timeout=120,
    max_retries=2,
)

search_tool = TavilySearch(
    max_results=MAX_SEARCH_RESULTS,
    topic="general",
)


@tool
def execute_python(code: str, timeout: int = 600) -> str:
    """Execute a Python script in the real project environment and return
    stdout/stderr.

    CRITICAL PATH RULE: paths reported by read_file/ls/glob (e.g. '/data/...',
    '/generated_code/...') are VIRTUAL paths rooted at /app on real disk. Code
    passed to this tool runs against the REAL filesystem, not the virtual one.
    You MUST prefix every such path with /app in code given to this tool:
        virtual '/data/EHR_SHOT/labels.csv'      -> real '/app/data/EHR_SHOT/labels.csv'
        virtual '/generated_code/ehr_models.py'  -> real '/app/generated_code/ehr_models.py'
    A bare '/data/...' or '/generated_code/...' path in code run by this tool
    will fail with FileNotFoundError or ModuleNotFoundError. If that happens,
    it means you forgot the /app prefix -- fix the path and retry, do not give up.
    """
    result = subprocess.run(
        ["python", "-c", code],
        cwd="/app",
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return (
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}\n\n"
        f"Exit code: {result.returncode}"
    )


# root_dir="/app" so the agent's virtual filesystem covers both /generated_code
# (existing models) and /data (train/test/val splits). Real disk equivalent of
# any virtual path is /app + that path.
benchmarking_agent = create_deep_agent(
    model=llm,
    tools=[search_tool, execute_python],
    backend=FilesystemBackend(root_dir="/app", virtual_mode=True),
)

messages = [
    SystemMessage(
        """
        You are an expert machine learning software engineer and biostatistician.

        Context:
        - Model implementations already exist under /generated_code. Do NOT
          reimplement or rewrite these models -- read the existing code and import
          from it.
        - Train/test/validation data splits are available under /data.
        - This is a fresh but non-empty working directory: /generated_code already
          contains files, /data already contains files. Use read_file / ls / glob
          to inspect them before writing anything.
        - If data format, label columns, or split structure are ambiguous, state
          your assumption explicitly in a code comment rather than guessing silently.

        PATH RULE (read carefully): read_file/ls/glob show VIRTUAL paths such as
        '/data/EHR_SHOT/labels.csv' or '/generated_code/ehr_models.py'. These are
        rooted at /app on the real filesystem. Any code you write -- both the test
        files you create AND the code you pass to execute_python -- must reference
        real paths, i.e. prefix every such path with /app:
            '/app/data/EHR_SHOT/labels.csv'
            '/app/generated_code/ehr_models.py'
        Do not write bare '/data/...' or '/generated_code/...' paths into Python
        code that will actually execute (open(), pd.read_csv(), imports via
        sys.path, etc.) -- those will fail on real disk. If you see
        FileNotFoundError or ModuleNotFoundError when running code, this is the
        most likely cause: fix the path prefix and retry rather than concluding
        the data or module is unavailable.

        You have an execute_python tool that runs real Python code in this
        project's environment. You MUST use it to actually run your test code
        against the real models and real data before finishing. Never claim a
        test works, or report metric values, without having actually observed
        them printed from a successful execute_python call. If execution fails,
        debug and retry -- do not fall back to describing what the code would
        do, and do not conclude the task is impossible without first trying the
        /app path prefix fix above.

        Your task is to write benchmarking test code, not new models. For each
        model implementation found in /generated_code, write a corresponding
        test module (e.g. test_<model_name>_benchmark.py) that:
        - loads the train/test/validation splits from /data (using real /app/data
          paths when actually executing)
        - trains or fits the model on the train split (using existing training
          utilities if present, otherwise a minimal fit call)
        - evaluates the model on the validation and/or test split
        - computes and reports standard classification benchmark metrics:
          accuracy, F1 score, precision, recall, AUROC, and any other metric
          appropriate to the model's task (e.g. Brier score for calibration,
          AUPRC for imbalanced classification)
        - uses clear assertions or printed output so results are inspectable
        - is well-documented and follows the existing code style in /generated_code

        Write all test files directly into /generated_code alongside the existing
        model code, so the module is self-contained.

        MANDATORY FINAL STEP: once you have actually run every model's benchmark
        test successfully and observed real metric values, write ALL results to
        a single file at /app/generated_code/benchmark_results.json (a real disk
        path, not a virtual one -- use execute_python or write_file with this
        exact real path) as a JSON object mapping each model name to a dict of
        its observed metrics, for example:
            {
              "count": {"accuracy": 0.71, "f1": 0.66, "auroc": 0.74, ...},
              "retain": {...},
              ...
            }
        Do not consider the task complete until this file exists on disk with
        real observed numbers in it. This file, not your chat summary, is the
        source of truth for results.
        """
    ),
    HumanMessage(
        f"""
        There are {NUMBER_OF_MODELS} models already implemented in /generated_code,
        originally based on this literature review: {response}.

        Inspect /generated_code to see what's already there, inspect /data to see
        what train/test/validation splits are available, then write benchmarking
        test code for each model as described above. Search the web if you need
        clarification on standard metric definitions or conventions for a
        particular task type.

        Run your test code with the execute_python tool -- remembering the /app
        path prefix rule -- until it actually succeeds and produces real metric
        output. Then write the complete results to
        /app/generated_code/benchmark_results.json as instructed. Only after that
        file exists with real numbers should you report the metric values you
        observed for each model in your final summary.
        """
    ),
]

for step in benchmarking_agent.stream({"messages": messages}, stream_mode="updates"):
    print(step)

# After the run, read back the authoritative results file directly from disk
# rather than trusting the agent's chat narration.
results_path = "/app/generated_code/benchmark_results.json"
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)
    print("\n=== FINAL BENCHMARK RESULTS (from disk) ===")
    print(json.dumps(results, indent=2))
else:
    print(f"\nWARNING: {results_path} was not created. The agent did not "
          f"complete a verified benchmark run. Check the stream above for "
          f"where it got stuck.")