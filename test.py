"""
Quick end-to-end smoke test: run a single logistic regression model that
classifies celiac disease from the EHRSHOT data, through the full
programming -> benchmarking -> reporting pipeline.

Uses number_of_models=1 and a literature_review string that pins the model
choice and the task, so the programming agent doesn't have to search for or
invent candidate models.
"""

from src.agents.programming_agent import programming_agent
from src.agents.benchmarking_agent import benchmarking_agent
from src.agents.draft_reporting_agent import reporting_agent  # agent-style version, see note above

LITERATURE_REVIEW = """
Task: binary classification of celiac disease diagnosis from EHRSHOT patient
records (single condition, single model, no comparison set).

Model: Logistic Regression.
Rationale: celiac diagnosis in EHR data is a well-studied binary outcome with
a mix of structured (labs, codes) and engineered features; logistic
regression is chosen as a fast, interpretable, well-calibrated baseline
before considering more complex models. Use scikit-learn's
LogisticRegression with L2 regularization, and standardize input features
before fitting.
"""

NUMBER_OF_MODELS = 1


def main():
    print("=== Running programming agent (logistic regression, celiac) ===")
    programming_trajectory = programming_agent(
        number_of_models=1,
        additional_context=[LITERATURE_REVIEW],
    )
    print(programming_trajectory["messages"][-1].content)

    print("\n=== Running benchmarking agent ===")
    benchmarking_trajectory = benchmarking_agent(
        number_of_models=1,
        additional_context=[LITERATURE_REVIEW],
    )
    print(benchmarking_trajectory["messages"][-1].content)

    print("\n=== Running reporting agent ===")
    reporting_trajectory = reporting_agent(
        number_of_models=1,
        literature_review=LITERATURE_REVIEW,
    )
    print(reporting_trajectory["messages"][-1].content)

    print("\nDone. Check /app/generated_code for model code, "
          "benchmark_results.json, and report.md")


if __name__ == "__main__":
    main()