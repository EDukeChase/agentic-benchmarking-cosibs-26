"""
FIX for patch_tolerate_extra_models.py: that earlier patch matched models
by EXACT slug (_slugify(model_name)), which incorrectly treats naming
variants of the SAME model as unrelated (e.g. literature candidate
"XGBoost (Extreme Gradient Boosting)" slugifies to
"xgboost_extreme_gradient_boosting", but the programming agent's actual
folder for that same model was just "xgboost" -- these were being wrongly
discarded as "extras" and could cause legitimate models to be dropped,
or the "too few models" RuntimeError to fire incorrectly.

This patch replaces exact-slug matching with FUZZY family matching (same
substring-based normalization used in aggregate_baseline_temp_sweep_v2.py),
so naming variants of the same model family are correctly recognized as
matches. Genuine duplicates (two generated folders mapping to the same
family) are still only counted once as a match; any second occurrence is
treated as an extra, same as a real unrequested model would be.

Requires patch_tolerate_extra_models.py to have already been applied.

Run once from /app:  python3 patch_fix_extra_model_matching.py
"""

with open("main.py") as f:
    content = f.read()

if "_normalize_model_family" in content:
    raise SystemExit("main.py already has fuzzy matching applied -- nothing to do.")

old_block = '''        model_code = collect_generated_models(prog_root)

        expected_slugs = {_slugify(c.model_name) for c in literature_result.candidates}
        matched_model_code = [m for m in model_code if m.model_name in expected_slugs]
        extra_model_code = [m for m in model_code if m.model_name not in expected_slugs]

        if extra_model_code:
            extra_names = [m.model_name for m in extra_model_code]
            print(
                f"NOTE: programming agent generated {len(extra_model_code)} extra "
                f"model(s) beyond the requested {number_of_models}: {extra_names}. "
                f"These are excluded from this run's benchmark/report, not the whole run."
            )
            audit_path = "/app/extra_models_audit.jsonl"
            with open(audit_path, "a") as audit_file:
                audit_file.write(json.dumps({
                    "run_id": run_id,
                    "expected_models": sorted(expected_slugs),
                    "extra_models_generated": extra_names,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }) + "\\n")

        model_code = matched_model_code

        if len(model_code) != number_of_models:
            raise RuntimeError(
                "Programming agent generated "
                f"{len(model_code)} usable model(s) matching the requested literature "
                f"candidates; expected {number_of_models}. "
                "Each model folder must contain model.py."
            )'''

new_block = '''        model_code = collect_generated_models(prog_root)

        expected_families = {_normalize_model_family(c.model_name) for c in literature_result.candidates}

        matched_model_code = []
        extra_model_code = []
        seen_families = set()
        for m in model_code:
            family = _normalize_model_family(m.model_name)
            if family in expected_families and family not in seen_families:
                matched_model_code.append(m)
                seen_families.add(family)
            else:
                extra_model_code.append(m)

        if extra_model_code:
            extra_names = [m.model_name for m in extra_model_code]
            print(
                f"NOTE: programming agent generated {len(extra_model_code)} extra "
                f"model(s) beyond the requested {number_of_models}: {extra_names}. "
                f"These are excluded from this run's benchmark/report, not the whole run."
            )
            audit_path = "/app/extra_models_audit.jsonl"
            with open(audit_path, "a") as audit_file:
                audit_file.write(json.dumps({
                    "run_id": run_id,
                    "expected_models": sorted(expected_families),
                    "extra_models_generated": extra_names,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }) + "\\n")

        model_code = matched_model_code

        if len(model_code) != number_of_models:
            raise RuntimeError(
                "Programming agent generated "
                f"{len(model_code)} usable model(s) matching the requested literature "
                f"candidates; expected {number_of_models}. "
                "Each model folder must contain model.py."
            )'''

assert old_block in content, (
    "expected tolerant-matching block (from patch_tolerate_extra_models.py) not found -- "
    "paste current main.py to get an exact patch."
)
content = content.replace(old_block, new_block, 1)

# Insert the fuzzy-matching helper right after the _slugify import line.
old_import = "from src.agents.programming_agent import build_programming_agent, run_programming_agent, collect_generated_models, _slugify"
helper = '''
import re as _re

_MODEL_FAMILY_PATTERNS = [
    ("logisticregression", "logistic_regression"),
    ("linearregression", "linear_regression"),
    ("randomforest", "random_forest"),
    ("extratrees", "extra_trees"),
    ("xgboost", "xgboost"),
    ("histgradientboosting", "gradient_boosting"),
    ("gradientboosting", "gradient_boosting"),
    ("lightgbm", "lightgbm"),
    ("catboost", "catboost"),
    ("adaboost", "adaboost"),
    ("decisiontree", "decision_tree"),
    ("supportvectormachine", "svm"),
    ("svc", "svm"),
    ("svm", "svm"),
    ("kneighbors", "knn"),
    ("knearestneighbor", "knn"),
    ("knn", "knn"),
    ("naivebayes", "naive_bayes"),
    ("ridge", "ridge_regression"),
    ("lasso", "lasso_regression"),
    ("elasticnet", "elastic_net"),
]


def _normalize_model_family(name: str) -> str:
    """Map a model_name to a canonical family, tolerant of naming variants
    (e.g. 'XGBoost' and 'XGBoost (Extreme Gradient Boosting)' -> 'xgboost').
    Falls back to a cleaned version of the raw name if nothing matches.
    """
    blob = _re.sub(r"[^a-z0-9]", "", name.lower())
    for pattern, canonical in _MODEL_FAMILY_PATTERNS:
        if pattern in blob:
            return canonical
    return blob or "unknown"
'''
assert old_import in content, "expected _slugify import line not found -- paste current main.py to verify"
content = content.replace(old_import, old_import + "\n" + helper, 1)

with open("main.py", "w") as f:
    f.write(content)

print("main.py patched: model matching now uses fuzzy family matching instead of exact slug matching.")
