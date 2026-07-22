"""
One-time patch: adds support for selecting specific named models (instead
of just "first N in list order") to main.py's literature-loading stage.

Run this once from /app:  python3 patch_main_for_model_names.py

Backward compatible: if BENCHMARK_CONDITION_JSON doesn't include
"model_names", behavior is unchanged (falls back to the existing
number_of_models positional slice).
"""

with open("main.py") as f:
    content = f.read()

# 1. Add the LiteratureReviewResult import (needed to build a filtered result)
old_import = "from src.base_literature import load_base_literature"
new_import = (
    "from src.base_literature import load_base_literature\n"
    "from src.schemas import LiteratureReviewResult"
)
assert old_import in content, "expected base_literature import line not found -- paste current main.py to verify"
content = content.replace(old_import, new_import, 1)

# 2. Replace the literature-loading stage to support model_names
old_stage = '''        stage = "literature review"
        print(f"Loading {number_of_models} models from the local base literature file...")
        # Restore these lines when Tavily use is permitted again:
        # lit_agent = build_literature_agent(
        #     max_search_results=max_search_results,
        #     llm_config=EXPERIMENT.literature_llm,
        # )
        # with stage_timeout(stage, timeout_seconds):
        #     literature_result = run_literature_review(
        #         lit_agent,
        #         num_models=number_of_models,
        #         system_prompt=LITERATURE_PROMPT,
        #     )
        literature_result = load_base_literature(num_models=number_of_models)'''

new_stage = '''        stage = "literature review"
        # Restore these lines when Tavily use is permitted again:
        # lit_agent = build_literature_agent(
        #     max_search_results=max_search_results,
        #     llm_config=EXPERIMENT.literature_llm,
        # )
        # with stage_timeout(stage, timeout_seconds):
        #     literature_result = run_literature_review(
        #         lit_agent,
        #         num_models=number_of_models,
        #         system_prompt=LITERATURE_PROMPT,
        #     )
        model_names = condition.get("model_names")
        if model_names:
            print(f"Loading specific models from the local base literature file: {model_names}")
            full_literature = load_base_literature(num_models=None)
            by_name = {c.model_name.lower(): c for c in full_literature.candidates}
            selected, missing = [], []
            for name in model_names:
                cand = by_name.get(name.lower())
                if cand is None:
                    missing.append(name)
                else:
                    selected.append(cand)
            if missing:
                raise RuntimeError(
                    f"Requested model_names not found in base literature: {missing}. "
                    f"Available: {[c.model_name for c in full_literature.candidates]}"
                )
            literature_result = LiteratureReviewResult(candidates=selected)
        else:
            print(f"Loading {number_of_models} models from the local base literature file...")
            literature_result = load_base_literature(num_models=number_of_models)'''

assert old_stage in content, "expected literature-loading stage block not found -- paste current main.py to verify"
content = content.replace(old_stage, new_stage, 1)

with open("main.py", "w") as f:
    f.write(content)

print("main.py patched successfully: model_names selection now supported.")
