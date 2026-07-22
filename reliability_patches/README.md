# Reliability Patches

Found and verified during a 40-run baseline temperature sweep. Each
script modifies main.py and/or the agent files in src/agents/ to fix a
specific failure mode. None of these change any prompt text.

Apply in this order (run once each, from /app):

1. `patch_benchmarking_recursion_limit.py`
   Raises the benchmarking agent's LangGraph recursion_limit from the
   default (~25) to 75. Fixes: agent runs out of self-correction steps
   while fixing its own virtual-vs-real path bugs, never writes
   benchmark_results.json.

2. `patch_rate_limit_backoff.py`
   Adds retry-with-backoff around each agent's .invoke() call in
   programming_agent.py, benchmarking_agent.py, and reporting_agent.py.
   Fixes: a transient Azure RateLimitError immediately kills the run
   instead of retrying.

3. `patch_tolerate_extra_models.py`
   Makes main.py tolerate the programming agent generating more models
   than requested, filtering extras out and logging them to
   extra_models_audit.jsonl instead of hard-failing the whole run.

4. `patch_fix_extra_model_matching.py`
   Required after #3. Fixes a bug in #3's matching logic: exact-slug
   matching incorrectly flagged naming variants of the same model
   (e.g. "xgboost" vs "xgboost_extreme_gradient_boosting") as unrelated
   extras. Switches to fuzzy family-based matching.

Each script is idempotent where possible and will tell you if it's
already been applied or if the expected code block isn't found (in
which case main.py/the agent files have likely changed since these were
written, and the script should be updated before reapplying).
