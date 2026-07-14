from .authentication import token_provider
from langchain_openai import ChatOpenAI
from src.schemas import GeneratedModel, BenchmarkResult, ModelCode, ReportNarrative, BenchmarkReport, ModelReportEntry
import json

SYSTEM_PROMPT = """
You are a biostatistics research scientist writing the results section of a benchmarking report.

For each candidate model, you will be given:
- rationale: why this model was selected during the literature review stage
- documentation: the implementing engineer's notes on implementation decisions, assumptions
  made where source documentation was incomplete, and known limitations
- accuracy, precision, recall, f1, brier: performance metrics on a held-out EHR test set

Write two things:

1. summary — for each model, in a short paragraph:
   - Report its aaccuracy, precision, recall, f1, and brier using only the numbers provided.
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

def merge_model_data(generated_models: list[GeneratedModel], model_code: list[ModelCode], results: list[BenchmarkResult]) -> list[ModelReportEntry]:
    results_by_name = {r.model_name: r for r in results}
    code_by_name = {c.model_name: c for c in model_code}

    merged = []
    for model in generated_models:
        result = results_by_name.get(model.model_name)
        code = code_by_name.get(model.model_name)
        if result is None:
            continue
        merged.append(ModelReportEntry(
            model_name=model.model_name,
            rationale=model.rationale,
            code=code.code,
            documentation=code.documentation,
            status=result.status,
            accuracy=result.accuracy,
            f1=result.f1,
            auroc=result.auroc,
        ))
    return merged

def build_reporting_agent():
    llm = ChatOpenAI(
        model = "gpt-5.4-mini",
        base_url = "https://bpsmar-ai-openai-1.openai.azure.com/openai/v1/",
        api_key = token_provider,
    )
    return llm.with_structured_output(ReportNarrative)

def build_report(structured_llm, generated_models: list[GeneratedModel], model_code: list[ModelCode], results: list[BenchmarkResult]) -> BenchmarkReport:
    entries = merge_model_data(generated_models, model_code, results)

    entries_json = json.dumps([e.model_dump(exclude={"code"}) for e in entries], indent=2)

    narrative = structured_llm.invoke(f"{SYSTEM_PROMPT}\n\nBenchmark results:\n{entries_json}")

    return BenchmarkReport(
        entries=entries,
        summary=narrative.summary,
        recommendations=narrative.recommendations,
    )