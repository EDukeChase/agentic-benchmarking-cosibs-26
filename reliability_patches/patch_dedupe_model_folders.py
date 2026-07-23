"""
Adds a dedup step to run_benchmarking_agent(), run BEFORE the agent is
invoked: any near-duplicate model folders (e.g. "random_forest" and
"random_forest_classifier" both existing for the same requested model)
are collapsed down to one folder per matched family, and the extras are
deleted from disk.

Why before invoke(), not after: the agent's own step/recursion budget was
being wasted sanity-checking and writing test files for folders it
shouldn't have needed to consider at all, leaving too little budget to
finish the genuinely requested models (observed in rep07: 5 folders
existed, only 3 got test files, and 2 of the 3 MISSING test files were
for actually-requested models). Removing duplicates before the agent
starts means it never sees them in the first place.

Genuinely unrequested extra models (e.g. LightGBM when only 3 specific
models were requested) are left untouched here -- that's a different,
already-handled case (see patch_tolerate_extra_models.py in main.py).
This patch only removes folders that are DUPLICATES of an already-kept,
requested model family.

Which duplicate is kept: folders are processed in sorted (alphabetical)
order: the first folder matching a given family is kept, later ones
mapping to the same family are deleted. This is simple and deterministic,
not necessarily "smartest" -- documented here as a known limitation.

No prompt text is touched.

Run once from /app:  python3 patch_dedupe_model_folders.py
"""

path = "src/agents/benchmarking_agent.py"

with open(path) as f:
    content = f.read()

if "_normalize_model_family" in content:
    raise SystemExit(f"{path} already has fuzzy-family logic -- nothing to do.")

# 1. Add `import shutil` alongside the existing imports.
old_imports = "import json\nimport subprocess\nimport os"
new_imports = "import json\nimport subprocess\nimport os\nimport shutil"
assert old_imports in content, "expected import block not found -- paste current benchmarking_agent.py to verify"
content = content.replace(old_imports, new_imports, 1)

# 2. Add the fuzzy-family helper (same patterns used in main.py / reporting_agent.py).
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
    """Map a folder/model name to a canonical family, tolerant of naming
    variants (e.g. 'random_forest' and 'random_forest_classifier' both ->
    'random_forest'). Falls back to a cleaned version of the raw name.
    """
    blob = _re.sub(r"[^a-z0-9]", "", name.lower())
    for pattern, canonical in _MODEL_FAMILY_PATTERNS:
        if pattern in blob:
            return canonical
    return blob or "unknown"
'''
old_anchor = "@tool\ndef execute_python"
assert old_anchor in content, "expected execute_python tool definition not found -- paste current file to verify"
content = content.replace(old_anchor, helper + "\n" + old_anchor, 1)

# 3. Insert the dedup step right after results_path is defined, before the
# system prompt is built -- so the agent never sees the duplicates at all.
old_block = '''    # the path for this specific run
    models_path = f"/generated_code/{run_id}"
    # the path where the agent should write the benchmarking results
    results_path = f"/generated_code/{run_id}/benchmark_results.json"

    # Use the configured prompt'''

new_block = '''    # the path for this specific run
    models_path = f"/generated_code/{run_id}"
    # the path where the agent should write the benchmarking results
    results_path = f"/generated_code/{run_id}/benchmark_results.json"

    # Collapse near-duplicate model folders (e.g. "random_forest" and
    # "random_forest_classifier" both existing for the same requested
    # model) down to one per family BEFORE the agent starts, so it never
    # wastes its step budget sanity-checking or testing folders it
    # shouldn't have needed to consider. Genuinely unrequested extras
    # (a different family entirely) are left untouched here.
    real_models_path_for_dedup = Path(f"/app{models_path}")
    expected_families_for_dedup = {
        _normalize_model_family(c.model_name) for c in literature_result.candidates
    }
    kept_by_family = {}
    duplicate_dirs = []
    if real_models_path_for_dedup.exists():
        for model_dir in sorted(real_models_path_for_dedup.iterdir()):
            if not model_dir.is_dir() or model_dir.name == "tmp":
                continue
            if not (model_dir / "model.py").exists():
                continue
            family = _normalize_model_family(model_dir.name)
            if family not in expected_families_for_dedup:
                continue
            if family in kept_by_family:
                duplicate_dirs.append(model_dir)
            else:
                kept_by_family[family] = model_dir
    if duplicate_dirs:
        print(
            f"NOTE: removing {len(duplicate_dirs)} duplicate model folder(s) "
            f"before benchmarking (keeping one per matched family): "
            f"{[d.name for d in duplicate_dirs]}"
        )
        for dup_dir in duplicate_dirs:
            shutil.rmtree(dup_dir)

    # Use the configured prompt'''

assert old_block in content, "expected models_path/results_path block not found -- paste current benchmarking_agent.py to verify"
content = content.replace(old_block, new_block, 1)

with open(path, "w") as f:
    f.write(content)

print(f"PATCHED {path}: duplicate model folders are now removed before the agent starts.")
