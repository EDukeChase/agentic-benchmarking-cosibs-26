from .authentication import token_provider
from langchain_openai import ChatOpenAI
from src.schemas import GeneratedModel, BenchmarkResult, ModelCode, ReportNarrative, BenchmarkReport, ModelReportEntry
import json
import re

SYSTEM_PROMPT = """
You are a biostatistics research scientist writing the results section of a benchmarking report.

For each candidate model, you will be given:
- rationale: why this model was selected during the literature review stage
- documentation: the implementing engineer's notes on implementation decisions, assumptions
  made where source documentation was incomplete, and known limitations
- accuracy, precision, recall, f1, auroc, brier: performance metrics on a held-out EHR test set

Write two things:

1. summary — for each model, in a short paragraph:
   - Report its accuracy, precision, recall, f1, auroc, and brier using only the numbers provided.
   - Connect the model's rationale and documented implementation choices (including any
     assumptions or limitations noted) to how it actually performed. For example, note if a
     documented limitation appears to explain a weaker score, or if a rationale's stated
     strength is reflected in the results.
   - Do not invent, estimate, or round metrics beyond what is given. Do not invent
     documentation, assumptions, or limitations that were not stated.

2. recommendations — recommend which model(s) to use for this prediction task, and justify it
   by weighing:
   - empirical performance (prioritize f1 and auroc and others over raw accuracy, given likely class
     imbalance in EHR outcome data)
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
        ))

    if skipped:
        raise RuntimeError(f"Cannot build report; missing artifacts for: {skipped}")

    return merged

def build_reporting_agent():
    llm = ChatOpenAI(
        model = "gpt-5.4-mini",
        base_url = "https://bpsmar-ai-openai-1.openai.azure.com/openai/v1/",
        api_key = token_provider,
    )
    return llm.with_structured_output(ReportNarrative)

def build_report(structured_llm, generated_models: list[GeneratedModel], model_code: list[ModelCode], results: list[BenchmarkResult], benchmark_scripts: dict[str, str]) -> BenchmarkReport:
    # merge all the model data into a single list of report entries
    entries = merge_model_data(generated_models, model_code, results, benchmark_scripts)

    # build json representation of the entries to pass to the LLM, excluding the code itself
    entries_json = json.dumps([e.model_dump(exclude={"code"}) for e in entries], indent=2)

    # invoke the LLM to generate the summary and recommendations for the report
    narrative = structured_llm.invoke(f"{SYSTEM_PROMPT}\n\nBenchmark results:\n{entries_json}")

    # return a BenchmarkReport object containing the entries and the narrative
    return BenchmarkReport(
        entries=entries,
        summary=narrative.summary,
        recommendations=narrative.recommendations,
    )
