from src.agents.literature_agent import literature_agent
from src.agents.programming_agent import programming_agent
from src.agents.benchmarking_agent import benchmarking_agent
from src.agents.reporting_agent import build_reporting_agent, build_report
# from src import tools

def main():
    number_of_models = 5

    literature_trajectory = literature_agent(number_of_models=number_of_models)
    programming_trajectory = programming_agent(number_of_models=number_of_models, additional_context=literature_trajectory)
    benchmarking_trajectory = benchmarking_agent(number_of_models=number_of_models, additional_context=literature_trajectory + programming_trajectory)
    # Call reporting agent and generate the benchmark report
    reporting_trajectory = build_reporting_agent()
    report = build_report(reporting_trajectory, generated_models, model_code, results)

if __name__ == "__main__":
    main()