"""
One-time patch: adds a "literature_prompt" override to main.py's literature
stage, matching the pattern already used for programming/benchmarking/
reporting (condition.get("<stage>_prompt", <DEFAULT>)). Currently the
literature stage is hardcoded to always use LITERATURE_PROMPT with no
override hook -- this adds one.

Run once from /app:  python3 patch_main_literature_prompt.py
"""

"""
One-time patch: adds a "literature_prompt" override to main.py's literature
stage, matching the pattern already used for programming/benchmarking/
reporting (condition.get("<stage>_prompt", <DEFAULT>)). Currently the
literature stage is hardcoded to always use LITERATURE_PROMPT with no
override hook -- this adds one.

Handles BOTH possible current states of main.py:
  (a) unpatched (plain live-search call at original indentation), or
  (b) already patched with the model_names bypass (patch_main_v3.py),
      where the same call is nested inside an "else:" branch with one
      extra level of indentation.
Whichever variant actually matches the file gets patched; the other is
just skipped.

Run once from /app:  python3 patch_main_literature_prompt.py
"""

with open("main.py") as f:
    content = f.read()

variants = [
    # (a) unpatched / no model_names branch -- 12-space base indent
    (
        '''            literature_result = run_literature_review(
                literature_agent,
                num_models=number_of_models,
                system_prompt=LITERATURE_PROMPT,
            )''',
        '''            literature_result = run_literature_review(
                literature_agent,
                num_models=number_of_models,
                system_prompt=condition.get("literature_prompt", LITERATURE_PROMPT),
            )''',
    ),
    # (b) model_names patch already applied -- nested in else branch, 16-space base indent
    (
        '''                literature_result = run_literature_review(
                    literature_agent,
                    num_models=number_of_models,
                    system_prompt=LITERATURE_PROMPT,
                )''',
        '''                literature_result = run_literature_review(
                    literature_agent,
                    num_models=number_of_models,
                    system_prompt=condition.get("literature_prompt", LITERATURE_PROMPT),
                )''',
    ),
]

applied = False
for old, new in variants:
    if old in content:
        content = content.replace(old, new, 1)
        applied = True
        break

if not applied:
    raise SystemExit(
        "Neither expected variant of the run_literature_review call was found. "
        "Paste current main.py to get an exact patch."
    )

with open("main.py", "w") as f:
    f.write(content)

print("main.py patched: literature stage now accepts a 'literature_prompt' override.")
