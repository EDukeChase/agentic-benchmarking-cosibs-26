from src.agents.reporting_agent import build_reporting_agent, build_report
from src import tools

def main():

    # Call reporting agent and generate the benchmark report
    reporting_llm = build_reporting_agent()
    report = build_report(reporting_llm, generated_models, model_code, results)

if __name__ == "__main__":
    main()