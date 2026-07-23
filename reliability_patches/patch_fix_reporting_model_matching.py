"""
One-time patch: fixes the same exact-slug matching bug in
reporting_agent.py that was already found and fixed in main.py
(see patch_fix_extra_model_matching.py).

merge_model_data() looked up code_by_name/results_by_name using
_slugify(model.model_name) -- an exact string match. When the literature
agent's candidate name ("Random Forest") differs from the actual
generated folder name ("random_forest_classifier"), or ("XGBoost") vs
("xgboost_extreme_gradient_boosting"), the exact match fails and the
model gets incorrectly reported as having "missing artifacts", even
though benchmarking succeeded and all the real files exist on disk.

This patch switches the lookup to the same fuzzy family-matching used in
main.py, so naming variants of the same model are correctly recognized.

No prompt text is touched.

Run once from /app:  python3 patch_fix_reporting_model_matching.py
"""

path = "src/agents/reporting_agent.py"

with open(path) as f:
    content = f.read()

if "_normalize_model_family" in content:
    raise SystemExit(f"{path} already has fuzzy matching applied -- nothing to do.")

# 1. Insert the same fuzzy-family helper used in main.py, right after _slugify.
old_slugify = '''def _slugify(name: str) -> str:
    """Match the canonical folder/model identifier used by the programming stage."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")'''

helper = '''def _slugify(name: str) -> str:
    """Match the canonical folder/model identifier used by the programming stage."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


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
    blob = re.sub(r"[^a-z0-9]", "", name.lower())
    for pattern, canonical in _MODEL_FAMILY_PATTERNS:
        if pattern in blob:
            return canonical
    return blob or "unknown"'''

assert old_slugify in content, "expected _slugify definition not found -- paste current reporting_agent.py to verify"
content = content.replace(old_slugify, helper, 1)

# 2. Replace the exact-slug lookup in merge_model_data with fuzzy family lookup.
old_merge = '''    code_by_name = {c.model_name: c for c in model_code}
    results_by_name = {r.model_name: r for r in results}

    # add a report entry for each generated model, if we have all the artifacts for it
    merged = []
    skipped = []
    for model in generated_models:
        artifact_name = _slugify(model.model_name)
        code_entry = code_by_name.get(artifact_name)
        result = results_by_name.get(artifact_name)
        benchmark_code = benchmark_scripts.get(artifact_name)'''

new_merge = '''    code_by_family = {_normalize_model_family(c.model_name): c for c in model_code}
    results_by_family = {_normalize_model_family(r.model_name): r for r in results}
    scripts_by_family = {_normalize_model_family(name): script for name, script in benchmark_scripts.items()}

    # add a report entry for each generated model, if we have all the artifacts for it
    merged = []
    skipped = []
    for model in generated_models:
        artifact_family = _normalize_model_family(model.model_name)
        code_entry = code_by_family.get(artifact_family)
        result = results_by_family.get(artifact_family)
        benchmark_code = scripts_by_family.get(artifact_family)'''

assert old_merge in content, "expected merge_model_data lookup block not found -- paste current reporting_agent.py to verify"
content = content.replace(old_merge, new_merge, 1)

with open(path, "w") as f:
    f.write(content)

print(f"PATCHED {path}: model matching now uses fuzzy family matching instead of exact slug matching.")
