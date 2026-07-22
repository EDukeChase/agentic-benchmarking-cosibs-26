"""
One-time patch: when BENCHMARK_CONDITION_JSON includes "model_names", this
bypasses the live literature-search agent entirely and constructs the
literature result directly and deterministically from a fixed local list.

Why bypass the search agent instead of just prompting it to pick these 3
models: a live search agent has no guarantee of returning the exact same
3 named models on every one of 40 runs. For a temperature sweep where
temperature is meant to be the ONLY manipulated variable, letting literature
selection vary run-to-run would silently introduce a second, uncontrolled
source of variation. Bypassing it removes that risk and also saves one
LLM call + real wall-clock time per run.

Backward compatible: without "model_names" in the condition JSON, behavior
is completely unchanged (goes through the live literature-search agent as
before).

Run once from /app:  python3 patch_main_for_model_names.py
"""

with open("main.py") as f:
    content = f.read()

# 1. Add the GeneratedModel/LiteratureReviewResult import
old_import = "from src.core.schemas import BenchmarkResult"
new_import = (
    "from src.core.schemas import BenchmarkResult, GeneratedModel, LiteratureReviewResult"
)
assert old_import in content, "expected schemas import line not found -- paste current main.py to verify"
content = content.replace(old_import, new_import, 1)

# 2. Add a fixed local candidate list, right after the EXPERIMENT config block
old_anchor = "LITERATURE_PROMPT = LITERATURE_SYSTEM_PROMPT"
new_anchor = '''LITERATURE_PROMPT = LITERATURE_SYSTEM_PROMPT

# Fixed candidates used only when BENCHMARK_CONDITION_JSON includes
# "model_names" -- bypasses the live literature-search agent for
# reproducibility across repeated runs (see patch docstring).
FIXED_LITERATURE_CANDIDATES = {
    "logistic regression": GeneratedModel(
        model_name="Logistic Regression",
        resource_name="scikit-learn LogisticRegression documentation",
        resource_link="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
        summary=(
            "Regularized logistic regression estimates the probability of a binary outcome "
            "using a linear decision function and a logistic link. It supports class weighting "
            "and several regularization and solver choices."
        ),
        rationale=(
            "Offers an interpretable and efficient classification baseline whose coefficients "
            "can help explain which EHR features contribute to a prediction."
        ),
    ),
    "random forest": GeneratedModel(
        model_name="Random Forest",
        resource_name="scikit-learn RandomForestClassifier documentation",
        resource_link="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
        summary=(
            "A random forest averages predictions from many decision trees trained on bootstrap "
            "samples with randomized feature subsets. It captures nonlinear effects and feature "
            "interactions without requiring feature scaling."
        ),
        rationale=(
            "Serves as a robust nonlinear benchmark for heterogeneous EHR features and provides "
            "feature-importance estimates for exploratory interpretation."
        ),
    ),
    "gradient boosting": GeneratedModel(
        model_name="Gradient Boosting",
        resource_name="scikit-learn HistGradientBoostingClassifier documentation",
        resource_link="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html",
        summary=(
            "Histogram-based gradient boosting builds decision trees sequentially so each new "
            "tree corrects errors made by the current ensemble. It supports nonlinear decision "
            "boundaries and is efficient on larger tabular datasets."
        ),
        rationale=(
            "Provides a strong tabular-data benchmark capable of modeling complex interactions "
            "among clinical variables while remaining practical to train."
        ),
    ),
}'''
assert old_anchor in content, "expected LITERATURE_PROMPT assignment line not found -- paste current main.py to verify"
content = content.replace(old_anchor, new_anchor, 1)

# 3. Replace the literature-loading stage to check for model_names first
old_stage = '''        stage = "literature review"
        print(
            f"Searching with {experiment.literature_llm.model} for "
            f"{number_of_models} candidate models..."
        )
        stage_started = time.perf_counter()
        literature_agent = build_literature_agent(
            llm_config=experiment.literature_llm,
        )
        with stage_timeout(stage, timeout_seconds):
            literature_result = run_literature_review(
                literature_agent,
                num_models=number_of_models,
                system_prompt=LITERATURE_PROMPT,
            )
        stage_timings[stage] = time.perf_counter() - stage_started'''

new_stage = '''        stage = "literature review"
        model_names = condition.get("model_names")
        stage_started = time.perf_counter()
        if model_names:
            print(f"Using fixed local literature candidates (search bypassed): {model_names}")
            selected, missing = [], []
            for name in model_names:
                cand = FIXED_LITERATURE_CANDIDATES.get(name.lower())
                if cand is None:
                    missing.append(name)
                else:
                    selected.append(cand)
            if missing:
                raise RuntimeError(
                    f"Requested model_names not found in FIXED_LITERATURE_CANDIDATES: {missing}. "
                    f"Available: {list(FIXED_LITERATURE_CANDIDATES.keys())}"
                )
            literature_result = LiteratureReviewResult(candidates=selected)
        else:
            print(
                f"Searching with {experiment.literature_llm.model} for "
                f"{number_of_models} candidate models..."
            )
            literature_agent = build_literature_agent(
                llm_config=experiment.literature_llm,
            )
            with stage_timeout(stage, timeout_seconds):
                literature_result = run_literature_review(
                    literature_agent,
                    num_models=number_of_models,
                    system_prompt=LITERATURE_PROMPT,
                )
        stage_timings[stage] = time.perf_counter() - stage_started'''

assert old_stage in content, "expected literature-loading stage block not found -- paste current main.py to verify"
content = content.replace(old_stage, new_stage, 1)

with open("main.py", "w") as f:
    f.write(content)

print("main.py patched successfully: model_names now bypasses live literature search.")
