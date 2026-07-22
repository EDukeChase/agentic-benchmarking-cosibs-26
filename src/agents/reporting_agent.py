from .authentication import token_provider
from langchain_openai import ChatOpenAI
from src.core.schemas import GeneratedModel, BenchmarkResult, ModelCode, ReportNarrative, BenchmarkReport, ModelReportEntry
import json
import re
from src.settings.config import LLMConfig, SelfConsistencyConfig
from src.settings.prompts import REPORTING_SYSTEM_PROMPT, SELF_CONSISTENCY_JUDGE_PROMPT

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


def _slugify(name: str) -> str:
    """Match the canonical folder/model identifier used by the programming stage."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

def merge_model_data(
    # we want to merge generated models, its code, results, and benchmark scripts into a single report entry for each model
    generated_models: list[GeneratedModel],
    model_code: list[ModelCode],
    results: list[BenchmarkResult],
    benchmark_scripts: dict[str, str],
) -> list[ModelReportEntry]:
    code_by_name = {c.model_name: c for c in model_code}
    results_by_name = {r.model_name: r for r in results}

    # add a report entry for each generated model, if we have all the artifacts for it
    merged = []
    skipped = []
    for model in generated_models:
        artifact_name = _slugify(model.model_name)
        code_entry = code_by_name.get(artifact_name)
        result = results_by_name.get(artifact_name)
        benchmark_code = benchmark_scripts.get(artifact_name)

        # if any of the artifacts are missing, skip this model and report it at the end
        if code_entry is None or result is None or benchmark_code is None:
            skipped.append(model.model_name)
            continue

        merged.append(ModelReportEntry(
            model_name=model.model_name,
            resource_name=model.resource_name,
            resource_link=model.resource_link,
            rationale=model.rationale,
            code=code_entry.code,
            documentation=code_entry.documentation,
            benchmark_code=benchmark_code,
            status="success",
            accuracy=result.accuracy,
            f1=result.f1,
            auroc=result.auroc,
            precision=result.precision,
            recall=result.recall,
            brier=result.brier,
            threshold=result.threshold,
        ))

    if skipped:
        raise RuntimeError(f"Cannot build report; missing artifacts for: {skipped}")

    return merged

def build_reporting_agent(llm_config: LLMConfig = LLMConfig()):
    llm = ChatOpenAI(
        model = llm_config.model,
        temperature = llm_config.temperature,
        base_url = "https://bpsmar-ai-openai-1.openai.azure.com/openai/v1/",
        api_key = token_provider,
        timeout = llm_config.timeout,
        max_retries = llm_config.max_retries,
    )
    return llm.with_structured_output(ReportNarrative)

def build_report(
    structured_llm,
    generated_models: list[GeneratedModel],
    model_code: list[ModelCode],
    results: list[BenchmarkResult],
    benchmark_scripts: dict[str, str],
    self_consistency: SelfConsistencyConfig = SelfConsistencyConfig(),
    system_prompt: str = REPORTING_SYSTEM_PROMPT,
    usage_sink: dict[str, int] | None = None,
    benchmark_assessment: str | None = None,
) -> BenchmarkReport:
    # merge all the model data into a single list of report entries
    entries = merge_model_data(generated_models, model_code, results, benchmark_scripts)

    # build json representation of the entries to pass to the LLM, excluding the code itself
    entries_json = json.dumps([e.model_dump(exclude={"code"}) for e in entries], indent=2)

    # invoke the LLM to generate the summary and recommendations for the report
    assessment_context = benchmark_assessment or "No separate benchmarking assessment was supplied."
    request = (
        f"{system_prompt}\n\nBenchmark results:\n{entries_json}\n\n"
        f"Benchmarking biostatistician's assessment:\n{assessment_context}"
    )
    try:
        from langchain_core.callbacks import UsageMetadataCallbackHandler
        usage_callback = UsageMetadataCallbackHandler()
        invoke_config = {"callbacks": [usage_callback]}
    except (ImportError, AttributeError):
        usage_callback = None
        invoke_config = None
    narratives = [
        _invoke_with_backoff(lambda: structured_llm.invoke(request, config=invoke_config))
        for _ in range(self_consistency.samples)
    ]

    if len(narratives) == 1:
        narrative = narratives[0]
    else:
        judge = build_reporting_agent(LLMConfig(
            model=self_consistency.model,
            temperature=self_consistency.temperature,
        ))
        candidates_json = json.dumps(
            [candidate.model_dump() for candidate in narratives], indent=2
        )
        narrative = judge.invoke(
            f"{SELF_CONSISTENCY_JUDGE_PROMPT}\n\n"
            f"Benchmark data:\n{entries_json}\n\n"
            f"Candidate narratives:\n{candidates_json}",
            config=invoke_config,
        )

    if usage_sink is not None and usage_callback is not None:
        for usage in usage_callback.usage_metadata.values():
            for key in usage_sink:
                usage_sink[key] += int(usage.get(key, 0) or 0)

    # return a BenchmarkReport object containing the entries and the narrative
    return BenchmarkReport(
        entries=entries,
        summary=narrative.summary,
        recommendations=narrative.recommendations,
    )
