"""
One-time patch: raises the LangGraph recursion_limit on the benchmarking
agent's .invoke() call. No prompt text is touched -- this is purely a
config change, consistent with keeping prompt wording unmodified per the
baseline parameters.

Rationale: repeated sweep failures show the agent writing a virtual-path
bug in its own sanity-check script (e.g. '/random_forest/model.py'
instead of '/app/random_forest/model.py'), then apparently running out of
steps while trying to self-correct before finishing the required
benchmark artifacts. No explicit recursion_limit was previously set,
so the agent was using LangGraph's default (commonly ~25 steps) --
raising this gives it more room to detect and fix its own mistake
without changing what it's told to do.

Run once from /app:  python3 patch_benchmarking_recursion_limit.py
"""

path = "src/agents/benchmarking_agent.py"

with open(path) as f:
    content = f.read()

old = '''    response = agent.invoke({"messages": [SystemMessage(system_prompt), HumanMessage(human_message)]})'''
new = '''    response = agent.invoke(
        {"messages": [SystemMessage(system_prompt), HumanMessage(human_message)]},
        config={"recursion_limit": 75},  # raised from LangGraph's default (~25) so the
                                          # agent has more room to self-correct (e.g. fix
                                          # its own virtual-vs-real path mistakes) before
                                          # giving up; no prompt text changed.
    )'''

assert old in content, "expected .invoke() call not found -- paste current benchmarking_agent.py to verify"
content = content.replace(old, new, 1)

with open(path, "w") as f:
    f.write(content)

print(f"{path} patched: recursion_limit raised to 75 on the benchmarking agent's invoke() call.")
