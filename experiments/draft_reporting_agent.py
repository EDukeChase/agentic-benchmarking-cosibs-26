from ..src.agents.authentication import token_provider
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage, SystemMessage, HumanMessage
from langchain.tools import tool
from langchain_tavily import TavilySearch
import subprocess
from openai import RateLimitError
import time

def call_with_token_backoff(agent, messages, max_attempts=6, base_delay=20):
    for attempt in range(1, max_attempts + 1):
        try:
            return agent.invoke({"messages": messages})
        except RateLimitError as e:
            if attempt == max_attempts:
                raise
            delay = base_delay * attempt  # linear backoff, tune as needed
            print(f"Rate limited (attempt {attempt}); waiting {delay}s...")
            time.sleep(delay)

@tool
def execute_python(code: str, timeout: int = 600) -> str:
    """Execute a Python script in the real project environment and return
    stdout/stderr. Useful for double-checking a metric or generating a plot
    to embed in the report, not for re-running full training.

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


def reporting_agent(
    number_of_models: int,
    literature_review: str,
    max_search_results: int = 10,
    additional_context: list[str] = [],
) -> list[AIMessage]:
    """Create a deep agent that writes a final report covering, for each model
    in /generated_code: the reasoning behind choosing/implementing it (from the
    literature review), the implementation itself, and its benchmarked accuracy
    metrics (from /generated_code/benchmark_results.json). Produces a single
    Markdown report file.
    """
    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        base_url="https://bpsmar-ai-openai-1.openai.azure.com/openai/v1/",
        api_key=token_provider,
        timeout=120,
        max_retries=2,
    )
    # search_tool = TavilySearch(
    #     max_results=max_search_results,
    #     topic="general",
    # )
    reporting_agent = create_deep_agent(
        model=llm,
        tools=[execute_python],
        backend=FilesystemBackend(root_dir="/app", virtual_mode=True),
    )
    messages = [
        SystemMessage(
            """
            You are an expert machine learning software engineer and biostatistician
            writing a technical report for a research audience.

            Context:
            - Model implementations already exist under /generated_code. Do NOT
            reimplement, rewrite, or "improve" this code -- your job is to explain
            and report on it, not to change it.
            - Benchmark results already exist at /generated_code/benchmark_results.json,
            a JSON object mapping each model name to a dict of its observed metrics
            (accuracy, F1, precision, recall, AUROC, etc.).
            - You will also be given the original literature review that motivated
            each model's selection.
            - This is a fresh but non-empty working directory: /generated_code already
            contains files. Use read_file / ls / glob to inspect everything before
            writing anything.

            PATH RULE (read carefully): read_file/ls/glob show VIRTUAL paths such as
            '/generated_code/ehr_models.py' or '/generated_code/benchmark_results.json'.
            These are rooted at /app on the real filesystem. If you use execute_python
            for any reason (e.g. to sanity-check a metric or render a chart), prefix
            every such path with /app:
                '/app/generated_code/ehr_models.py'
                '/app/generated_code/benchmark_results.json'
            Do not write bare '/generated_code/...' paths into code that will actually
            execute -- those will fail on real disk.

            Your task is to write a single, well-organized Markdown report, not new
            model code and not new tests. For each model found in /generated_code,
            the report must include:
            - Rationale: why this model was chosen, drawn from the literature review
              provided to you -- what problem it addresses, what prior work motivates
              it, and any tradeoffs noted.
            - Implementation: a concise description of how the model is implemented,
              including the key architecture/training details, with the most relevant
              code excerpts included in fenced code blocks (do not paste entire files
              verbatim if they are long -- excerpt the parts that matter and describe
              the rest in prose).
            - Results: the model's observed metrics from benchmark_results.json,
              presented in a Markdown table, with a short interpretation (e.g. what
              the AUROC/F1 imply about the model's performance on this task, and how
              it compares to the other models in the report).

            The report should open with a brief executive summary comparing all
            models at a glance (a single combined metrics table is ideal), and close
            with a short discussion of limitations and any recommended next steps.

            Never report a metric value you have not actually read from
            benchmark_results.json. If a model in /generated_code has no
            corresponding entry in benchmark_results.json, say so explicitly in the
            report rather than inventing or estimating a number. If something about
            a model's rationale is not covered by the literature review, say so
            explicitly rather than inventing a justification.

            MANDATORY FINAL STEP: write the complete report to
            /app/generated_code/report.md (a real disk path, not a virtual one --
            use execute_python or write_file with this exact real path). Do not
            consider the task complete until this file exists on disk. This file,
            not your chat summary, is the deliverable.
            """
        ),
        HumanMessage(
            f"""
            There are {number_of_models} models already implemented in /generated_code,
            originally based on this literature review:

            {literature_review}

            Benchmark results for these models are available at
            /generated_code/benchmark_results.json.

            Inspect /generated_code to see the implementations and the benchmark
            results file, then write the full report as described above -- covering
            the reasoning behind each model, its implementation, and its accuracy
            metrics. Search the web if you need additional context on a technique
            or metric mentioned in the literature review.

            Once the report is complete, write it to /app/generated_code/report.md
            as instructed, then summarize the report's key findings in your final
            response.
            """
        ),
    ]
    for context in additional_context:
        messages.append(HumanMessage(context))

    trajectory = call_with_token_backoff(reporting_agent, messages)
    return trajectory