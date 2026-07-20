from .authentication import token_provider
from langchain_openai import ChatOpenAI
from src.schemas import GeneratedModel, BenchmarkResult, ModelCode, ReportNarrative, BenchmarkReport, ModelReportEntry
import json
import re
from src.config import LLMConfig, SelfConsistencyConfig
from src.prompts import REPORTING_SYSTEM_PROMPT, SELF_CONSISTENCY_JUDGE_PROMPT

SYSTEM_PROMPT = """
You are a biostatistics research scientist writing the results section of a benchmarking report.

For each candidate model, you will be given:
- rationale: why this model was selected during the literature review stage
- documentation: the implementing engineer's notes on implementation decisions, assumptions
  made where source documentation was incomplete, and known limitations
- accuracy, precision, recall, f1, auroc, brier: performance metrics on a held-out EHR test set

Write two things:

1. summary — for each model, in a short paragraph:
   - Report AUROC, F1, recall, precision, and Brier score first. These are the
     primary metrics for this imbalanced diagnosis task.
   - Report accuracy last as a secondary descriptive metric.
   - Connect the model's rationale and documented implementation choices (including any
     assumptions or limitations noted) to how it actually performed. For example, note if a
     documented limitation appears to explain a weaker score, or if a rationale's stated
     strength is reflected in the results.
   - Do not invent, estimate, or round metrics beyond what is given. Do not invent
     documentation, assumptions, or limitations that were not stated.

2. recommendations — recommend which model(s) to use for this prediction task, and justify it
   by weighing:
   - empirical performance using AUROC, F1, recall, precision, and Brier score
     as primary metrics
   - accuracy only as a secondary metric; never recommend a model mainly for
     high accuracy when its F1 or recall is zero
   - the documented assumptions and limitations of each implementation, since a model with
     strong metrics but significant undocumented-source assumptions may be less trustworthy
     than one with clearly documented, minor limitations

Be concise and factual — this is a benchmark report, not a persuasive essay. Do not fabricate
any detail not present in the provided data.
"""


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
) -> BenchmarkReport:
    # merge all the model data into a single list of report entries
    entries = merge_model_data(generated_models, model_code, results, benchmark_scripts)

    # build json representation of the entries to pass to the LLM, excluding the code itself
    entries_json = json.dumps([e.model_dump(exclude={"code"}) for e in entries], indent=2)

    # invoke the LLM to generate the summary and recommendations for the report
    request = f"{system_prompt}\n\nBenchmark results:\n{entries_json}"
    try:
        from langchain_core.callbacks import UsageMetadataCallbackHandler
        usage_callback = UsageMetadataCallbackHandler()
        invoke_config = {"callbacks": [usage_callback]}
    except (ImportError, AttributeError):
        usage_callback = None
        invoke_config = None
    narratives = [structured_llm.invoke(request, config=invoke_config) for _ in range(self_consistency.samples)]

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
