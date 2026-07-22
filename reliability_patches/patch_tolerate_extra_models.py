"""
One-time patch: instead of hard-failing a run when the programming agent
generates MORE models than requested, this filters down to only the
models matching the literature agent's selected candidates, and logs any
extras to an audit file for tracking -- rather than discarding the whole
run's completed work.

A run still fails if it has FEWER matching models than requested (that's
a real problem, unchanged). Only the "too many" case is now tolerated.

Also creates/appends to /app/extra_models_audit.jsonl -- one JSON line
per affected run, recording which extra model(s) were generated and
discarded, so this can be reported as its own finding (how often, and
what, the agent adds beyond what was asked).

No prompt text is touched -- this is a validation-logic change only.

Run once from /app:  python3 patch_tolerate_extra_models.py
"""

with open("main.py") as f:
    content = f.read()

# 1. Import _slugify from programming_agent so main.py can match generated
# model folder names against the literature agent's expected candidate names.
old_import = "from src.agents.programming_agent import build_programming_agent, run_programming_agent, collect_generated_models"
new_import = "from src.agents.programming_agent import build_programming_agent, run_programming_agent, collect_generated_models, _slugify"
assert old_import in content, "expected programming_agent import line not found -- paste current main.py to verify"
content = content.replace(old_import, new_import, 1)

# 2. Replace the strict model-count validation with a tolerant version.
old_validation = '''        model_code = collect_generated_models(prog_root)
        if len(model_code) != number_of_models:
            raise RuntimeError(
                "Programming agent generated "
                f"{len(model_code)} usable model(s); expected {number_of_models}. "
                "Each model folder must contain model.py."
            )'''

new_validation = '''        model_code = collect_generated_models(prog_root)

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

assert old_validation in content, "expected validation block not found -- paste current main.py to verify"
content = content.replace(old_validation, new_validation, 1)

with open("main.py", "w") as f:
    f.write(content)

print("main.py patched: extra models are now filtered out and logged, not treated as a hard failure.")
print("Audit trail will accumulate in /app/extra_models_audit.jsonl")
