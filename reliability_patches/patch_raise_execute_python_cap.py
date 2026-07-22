"""
One-time patch: raises the hard cap on individual execute_python tool
calls in benchmarking_agent.py from 480 seconds to 900 seconds.

This cap was added upstream (not by any of our earlier patches) and
silently overrides whatever timeout value gets passed in -- meaning even
though BENCHMARK_STAGE_TIMEOUT_SECONDS controls the OUTER stage budget
(currently 1500s), any single execute_python call within that stage was
still being clamped to at most 480s regardless. A long-running benchmark
script (training 3 models, writing predictions.json, etc. in one call)
could hit this inner cap well before the outer stage timeout, causing a
truncated/failed execution that looks like a normal stage timeout but
is actually this separate, tighter limit.

900s gives real headroom under the 1500s outer stage budget while still
bounding any single runaway call.

Run once from /app:  python3 patch_raise_execute_python_cap.py
"""

path = "src/agents/benchmarking_agent.py"

with open(path) as f:
    content = f.read()

old = "    timeout = min(max(int(timeout), 1), 480)"
new = "    timeout = min(max(int(timeout), 1), 900)  # raised from 480 -- see patch docstring"

if new in content:
    print(f"SKIP: {path} already has the raised cap, not re-patching.")
elif old in content:
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print(f"PATCHED {path}: execute_python timeout cap raised from 480s to 900s.")
else:
    raise SystemExit(
        f"Expected timeout-cap line not found in {path} -- "
        "paste current file to get an exact patch."
    )
