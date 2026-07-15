import uuid
from src.agents.literature_agent import build_literature_agent, run_literature_review
from src.agents.programming_agent import build_programming_agent, run_programming_agent, collect_generated_models
from src.agents.benchmarking_agent import build_benchmarking_agent, run_benchmarking_agent
from src.benchmark_tools import collect_benchmark_results, collect_benchmark_scripts
from src.agents.reporting_agent import build_reporting_agent, build_report
from src.markdown_report import save_markdown
from src.schemas import BenchmarkResult

def main():
    # generate a unique run ID for this benchmarking session
    run_id = uuid.uuid4().hex[:8]

    # define the number of models to benchmark and the maximum number of search results to consider
    number_of_models = 1
    max_search_results = 1

    print(f"Starting new run with ID: {run_id}")
    
    print(f"Running literature review for {number_of_models} models with max {max_search_results} search results...")
    lit_agent = build_literature_agent(max_search_results=max_search_results)
    literature_result = run_literature_review(lit_agent, num_models=number_of_models)

    print(f"Running programming agent to generate code for the models...")
    prog_root = f"/app/generated_code/{run_id}"
    prog_agent = build_programming_agent(prog_root)
    run_programming_agent(prog_agent, literature_result)
    model_code = collect_generated_models(prog_root)

    print(f"Running benchmarking agent to evaluate the generated models...")
    bench_agent = build_benchmarking_agent()

    print(f"Initialized benchmarking agent for run {run_id}")
    run_benchmarking_agent(bench_agent, run_id, literature_result)

    print(f"Collecting benchmark results and scripts for run {run_id}...")

    raw_results = collect_benchmark_results(run_id)
    results = [
        BenchmarkResult(model_name=name, **metrics)
        for name, metrics in raw_results.items()
    ]
    benchmark_scripts = collect_benchmark_scripts(run_id)

    print(f"Run {run_id} completed. Benchmark results and scripts collected.")

    print(f"Building benchmark report for run {run_id}...")

    reporting_llm = build_reporting_agent()
    report = build_report(
        reporting_llm,
        literature_result.candidates,
        model_code,
        results,
        benchmark_scripts,
    )

    report_path = f"/app/generated_code/{run_id}/report.json"
    with open(report_path, "w") as f:
        f.write(report.model_dump_json(indent=2))

    markdown_report_path = f"/app/generated_code/{run_id}/report.md"
    save_markdown(report, markdown_report_path)

    print(f"Report written to {report_path} and {markdown_report_path}")
    print(f"Run {run_id} completed. Benchmark results and scripts collected.")

if __name__ == "__main__":
    main()
