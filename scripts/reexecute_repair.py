"""Re-execute a run; if its code fails to run, let a repair agent try to make it run (draft, first version).

    python3 scripts/reexecute_repair.py --run-dir <run> --tool-image ssa-study:<tag> --brief <brief.md>

1. The first attempt is scripts/reexecute_run.py with the run's code unchanged. If it ran, that is the result.
2. If it failed with an error the code or its surroundings can explain (not a time limit or a memory kill), a
   headless Claude Code session is started on a COPY of the code, with a fixed written brief (hash recorded).
   The session can read, search, edit and write files in that copy and nothing else: no shell, no web, no
   other tools. It cannot run anything. After it stops, this script runs the edited copy in the same fresh
   container, up to `--max-cycles` times, feeding back the new error text.
3. This script, not the agent, computes the diff against the original code and records the agent's own note
   (CHANGE_NOTE.md), the model, tokens, cost and time of every session, and the brief's SHA-256.
The first attempt always stays on record as failed; a repaired execution is stored and labelled separately.
Results: <result-root>/<run id>/exec<n>/repair.json and exec<n>_repair<k>/ (one folder per repaired attempt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import reexecute_run  # noqa: E402

TOOLS = "Read,Edit,Write,Glob,Grep"
NOT_REPAIRABLE = ("time limit", "out of memory")
TASK_FIRST = ("Follow the brief. The error output of the failed attempt is in ERROR_LOG.txt, the original package list in PACKAGES.txt "
              "and the packages available now in PACKAGES_NOW.txt. The code is in the code/ folder. Make the smallest change that lets "
              "it run, write CHANGE_NOTE.md as the brief says, then stop.")
TASK_AGAIN = ("The code was run again after your last change and it failed again. The new error output is in ERROR_LOG.txt and your "
              "earlier note is in CHANGE_NOTE.md. Follow the brief again: make the smallest further change, update CHANGE_NOTE.md, then stop.")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def prepare_workspace(workspace: Path, original_code: Path, error_text: str, recorded_packages: str, current_packages: str,
                      data_files: str = "") -> None:
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    shutil.copytree(original_code, workspace / "code", ignore=lambda folder, names: [n for n in names if (Path(folder) / n).is_symlink()])
    (workspace / "PACKAGES.txt").write_text(recorded_packages)
    (workspace / "PACKAGES_NOW.txt").write_text(current_packages)
    (workspace / "ERROR_LOG.txt").write_text(error_text)
    (workspace / "DATA_FILES.txt").write_text(data_files)


def call_agent(workspace: Path, brief_file: Path, task: str, minutes: float, max_turns: int, budget_usd: float, model: str | None) -> dict:
    """One headless Claude Code session in `workspace`. Returns the session record (never raises)."""
    command = ["claude", "-p", task, "--append-system-prompt-file", str(brief_file), "--tools", TOOLS, "--disable-slash-commands",
               "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence", "--max-turns", str(max_turns),
               "--max-budget-usd", str(budget_usd), "--permission-mode", "acceptEdits", "--output-format", "json"]
    if model:
        command += ["--model", model]
    env = dict(os.environ, CLAUDE_CODE_DISABLE_AUTO_MEMORY="1")
    started = time.monotonic()
    try:
        done = subprocess.run(command, cwd=workspace, env=env, capture_output=True, text=True, timeout=minutes * 60)
        timed_out, stdout, stderr = False, done.stdout, done.stderr
    except subprocess.TimeoutExpired as error:
        timed_out, stdout, stderr = True, (error.stdout or ""), (error.stderr or "")
        stdout = stdout if isinstance(stdout, str) else stdout.decode(errors="replace")
        stderr = stderr if isinstance(stderr, str) else stderr.decode(errors="replace")
    record = {"seconds": round(time.monotonic() - started, 1), "timed_out": timed_out, "tools": TOOLS, "stderr_tail": stderr[-500:]}
    try:
        data = json.loads(stdout)
    except ValueError:
        record.update(parsed=False, stdout_tail=stdout[-500:])
        return record
    usage = data.get("usage") or {}
    record.update(parsed=True, is_error=data.get("is_error"), num_turns=data.get("num_turns"), cost_usd_list_price=data.get("total_cost_usd"),
                  models=list((data.get("modelUsage") or {}).keys()), session_id=data.get("session_id"),
                  tokens={"input": usage.get("input_tokens"), "cache_write": usage.get("cache_creation_input_tokens"),
                          "cache_read": usage.get("cache_read_input_tokens"), "output": usage.get("output_tokens")},
                  stop_reason=data.get("stop_reason"), final_message=(data.get("result") or "")[:600])
    return record


def data_file_names(data_dir: Path | None) -> str:
    """The names of the files in the data folder, nothing else (the agent never sees data)."""
    try:
        folder = data_dir or reexecute_run.ssa_run.data_mount_for(reexecute_run.ssa_run.image_meta(reexecute_run.ssa_run.DATASET))
        return "\n".join(sorted(p.name for p in Path(folder).iterdir())) + "\n"
    except (OSError, reexecute_run.ssa_run.SetupError):
        return ""


def make_diff(original: Path, edited: Path) -> str:
    done = subprocess.run(["diff", "-ruN", "--exclude=__pycache__", str(original), str(edited)], capture_output=True, text=True)
    return done.stdout


def read_text(path: Path, limit: int = 20000) -> str:
    try:
        return Path(path).read_text(errors="replace")[-limit:]
    except OSError:
        return ""


def main(argv: list[str] | None = None, execute=reexecute_run.main, agent=call_agent) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--execution", type=int, default=1)
    parser.add_argument("--tool-image", required=True)
    parser.add_argument("--brief", required=True, type=Path, help="the approved brief (a file outside the repo); its SHA-256 is recorded")
    parser.add_argument("--result-root", type=Path, default=reexecute_run.RESULT_ROOT)
    parser.add_argument("--data-dir", type=Path, default=None, help="default: the mounted data image (only its file names go to the agent)")
    parser.add_argument("--cpuset", default="10-18")
    parser.add_argument("--memory", default="11g")
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--agent-minutes", type=float, default=15.0)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--budget-usd", type=float, default=2.0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-hours", type=float, default=4.0)
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    run_id = run_dir.name
    brief_text = read_text(args.brief, limit=10**6)
    if not brief_text.strip():
        print(f"Not started: the brief {args.brief} is missing or empty", file=sys.stderr)
        return 3
    brief_sha = sha256_text(brief_text)
    base = ["--run-dir", str(run_dir), "--tool-image", args.tool_image, "--result-root", str(args.result_root),
            "--cpuset", args.cpuset, "--memory", args.memory, "--max-hours", str(args.max_hours)]
    if args.data_dir:
        base += ["--data-dir", str(args.data_dir)]
    exec_dir = args.result_root / run_id / f"exec{args.execution}"
    if not (exec_dir / "result.json").exists():
        code = execute(base + ["--execution", str(args.execution)])
        if code == 3:
            return 3
    first = reexecute_run.read_json(exec_dir / "result.json") or {}
    summary = {"run_id": run_id, "execution": args.execution, "brief_sha256": brief_sha, "brief_file_name": args.brief.name,
               "first_attempt": first.get("first_attempt"), "failure_cause_guess": first.get("failure_cause_guess"),
               "repair_attempted": False, "repaired_and_ran": False, "cycles": [], "limits": {
                   "max_cycles": args.max_cycles, "agent_minutes": args.agent_minutes, "max_turns": args.max_turns, "budget_usd": args.budget_usd}}
    if first.get("first_attempt") != "failed":
        summary["note"] = "the first attempt ran; no repair needed" if first else "no first-attempt result found"
    elif first.get("failure_cause_guess") in NOT_REPAIRABLE:
        summary["note"] = f"not repairable by the brief ({first.get('failure_cause_guess')}); recorded as failed"
    else:
        summary["repair_attempted"] = True
        workspace = exec_dir / "repair_workspace"
        original_code = run_dir / "out" / "code"
        recorded = read_text(run_dir / "records" / "packages.txt")
        current = "\n".join(reexecute_run.installed_packages(first.get("environment_image") or args.tool_image))
        error_text = read_text(exec_dir / "reexec.log", 6000)
        prepare_workspace(workspace, original_code, error_text, recorded, current, data_file_names(args.data_dir))
        for cycle in range(1, args.max_cycles + 1):
            session = agent(workspace, args.brief, TASK_FIRST if cycle == 1 else TASK_AGAIN, args.agent_minutes, args.max_turns,
                            args.budget_usd, args.model)
            diff = make_diff(original_code, workspace / "code")
            subdir = f"exec{args.execution}_repair{cycle}"
            cycle_dir = args.result_root / run_id / subdir
            cycle_dir.mkdir(parents=True, exist_ok=True)
            (cycle_dir / "changes.diff").write_text(diff)
            if (workspace / "CHANGE_NOTE.md").exists():
                shutil.copy2(workspace / "CHANGE_NOTE.md", cycle_dir / "change_note.md")
            (cycle_dir / "agent_session.json").write_text(json.dumps({**session, "brief_sha256": brief_sha, "cycle": cycle}, indent=1))
            entry = {"cycle": cycle, "session": {k: session.get(k) for k in ("seconds", "timed_out", "num_turns", "cost_usd_list_price", "tokens", "models", "is_error")},
                     "diff_lines": len(diff.splitlines()), "result_folder": subdir, "ran": None}
            if not diff.strip():
                entry["note"] = "the agent changed nothing; stopping"
                summary["cycles"].append(entry)
                break
            code = execute(base + ["--execution", str(args.execution), "--code-dir", str(workspace / "code"), "--result-subdir", subdir, "--repaired"])
            result = reexecute_run.read_json(cycle_dir / "result.json") or {}
            entry["ran"] = result.get("first_attempt") == "ran"
            entry["all_within_tolerance"] = result.get("all_within_tolerance")
            summary["cycles"].append(entry)
            if entry["ran"]:
                summary["repaired_and_ran"] = True
                break
            (workspace / "ERROR_LOG.txt").write_text(read_text(cycle_dir / "reexec.log", 6000))
        summary["total_cost_usd_list_price"] = round(sum((c["session"].get("cost_usd_list_price") or 0) for c in summary["cycles"]), 4)
    summary["note"] = summary.get("note") or ("repaired and ran" if summary["repaired_and_ran"] else "could not be repaired")
    (exec_dir / "repair.json").write_text(json.dumps(summary, indent=1))
    print(f"{run_id} execution {args.execution}: first attempt {summary['first_attempt']}; {summary['note']}")
    return 0 if summary["first_attempt"] == "ran" or summary["repaired_and_ran"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
