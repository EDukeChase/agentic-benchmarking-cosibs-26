"""
One-time patch: adds retry-with-exponential-backoff around each agent's
.invoke() call, so a transient RateLimitError from Azure mid-run doesn't
immediately kill the whole run.

Applies to:
  - src/agents/programming_agent.py
  - src/agents/benchmarking_agent.py
  - src/agents/reporting_agent.py  (patches structured_llm.invoke calls)

No prompt text is touched anywhere -- this is purely retry/backoff logic
wrapped around the existing invoke() calls, consistent with keeping
prompt wording unmodified per the baseline parameters.

Run once from /app:  python3 patch_rate_limit_backoff.py
"""

import re

BACKOFF_HELPER = '''
from openai import RateLimitError as _RateLimitError
import time as _time

def _invoke_with_backoff(callable_fn, max_attempts=6, base_delay=20):
    """Retry callable_fn() on RateLimitError with linear backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            return callable_fn()
        except _RateLimitError:
            if attempt == max_attempts:
                raise
            delay = base_delay * attempt
            print(f"Rate limited (attempt {attempt}/{max_attempts}); waiting {delay}s before retrying...")
            _time.sleep(delay)
'''


def patch_file(path, old_invoke, new_invoke, label):
    with open(path) as f:
        content = f.read()

    if "_invoke_with_backoff" in content:
        print(f"SKIP {label}: backoff helper already present, not re-patching.")
        return

    # Insert the helper right after the last top-level import line.
    import_lines = [l for l in content.splitlines() if l.startswith("import ") or l.startswith("from ")]
    if not import_lines:
        raise SystemExit(f"{label}: no import lines found to anchor the patch -- paste current file to verify")
    last_import = import_lines[-1]
    content = content.replace(last_import, last_import + "\n" + BACKOFF_HELPER, 1)

    if old_invoke not in content:
        raise SystemExit(
            f"{label}: expected invoke() call not found after inserting helper -- "
            f"paste current {path} to get an exact patch."
        )
    content = content.replace(old_invoke, new_invoke, 1)

    with open(path, "w") as f:
        f.write(content)
    print(f"PATCHED {label}: backoff wrapper added around its invoke() call.")


# --- programming_agent.py ---
patch_file(
    "src/agents/programming_agent.py",
    old_invoke='''    return agent.invoke({"messages": [SystemMessage(system_prompt), HumanMessage(human_message)]})''',
    new_invoke='''    return _invoke_with_backoff(
        lambda: agent.invoke({"messages": [SystemMessage(system_prompt), HumanMessage(human_message)]})
    )''',
    label="programming_agent.py",
)

# --- benchmarking_agent.py ---
patch_file(
    "src/agents/benchmarking_agent.py",
    old_invoke='''    response = agent.invoke(
        {"messages": [SystemMessage(system_prompt), HumanMessage(human_message)]},
        config={"recursion_limit": 75},  # raised from LangGraph's default (~25) so the
                                          # agent has more room to self-correct (e.g. fix
                                          # its own virtual-vs-real path mistakes) before
                                          # giving up; no prompt text changed.
    )''',
    new_invoke='''    response = _invoke_with_backoff(
        lambda: agent.invoke(
            {"messages": [SystemMessage(system_prompt), HumanMessage(human_message)]},
            config={"recursion_limit": 75},
        )
    )''',
    label="benchmarking_agent.py",
)

# --- reporting_agent.py ---
# reporting_agent.py builds a list via a list-comprehension .invoke() call
# (self_consistency.samples), so it's patched separately below rather than
# via the generic patch_file() helper.
with open("src/agents/reporting_agent.py") as f:
    reporting_content = f.read()

if "_invoke_with_backoff" in reporting_content:
    print("SKIP reporting_agent.py: backoff helper already present, not re-patching.")
else:
    import_lines = [l for l in reporting_content.splitlines() if l.startswith("import ") or l.startswith("from ")]
    last_import = import_lines[-1]
    reporting_content = reporting_content.replace(last_import, last_import + "\n" + BACKOFF_HELPER, 1)

    old_narratives = '''    narratives = [structured_llm.invoke(request, config=invoke_config) for _ in range(self_consistency.samples)]'''
    new_narratives = '''    narratives = [
        _invoke_with_backoff(lambda: structured_llm.invoke(request, config=invoke_config))
        for _ in range(self_consistency.samples)
    ]'''

    if old_narratives not in reporting_content:
        raise SystemExit(
            "reporting_agent.py: expected narratives list-comprehension not found after "
            "inserting helper -- paste current reporting_agent.py to get an exact patch."
        )
    reporting_content = reporting_content.replace(old_narratives, new_narratives, 1)

    with open("src/agents/reporting_agent.py", "w") as f:
        f.write(reporting_content)
    print("PATCHED reporting_agent.py: backoff wrapper added around its invoke() call.")

print("\nDone. All three agent files now retry on RateLimitError with linear backoff before failing.")
