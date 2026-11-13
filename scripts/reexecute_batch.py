"""Re-execute every counted run, in the order the runs finished (draft).

    python3 scripts/reexecute_batch.py --tool-image ssa-study:<tag> --dry-run     # show the plan
    python3 scripts/reexecute_batch.py --tool-image ssa-study:<tag>               # run it

Plan: execution 1 for every counted run (a study run whose pipeline finished with success) in finish order, then
execution 2 for the runs of the subsample. The subsample rule is fixed in advance and uses no result:
    k = min(n, max(3, ceil(0.10 * n)))     n = number of counted runs
    the k runs with the lowest score, score = SHA-256("<seed>|SSA|<run id>") read as an integer, seed 5960.
A run that already has a result for an execution is skipped, so the batch can be stopped and started again.
It waits while an SSA study container is running on the same CPUs, and stops between two executions when the
stop file exists. Each execution is scripts/reexecute_run.py (the run's own code, unchanged).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import reexecute_run, ssa_run  # noqa: E402

SEED = 5960
FRACTION = 0.10
FLOOR = 3


def counted_runs(runs_dir: Path) -> list[tuple[str, str]]:
    """[(finished stamp, run id)] of study runs whose pipeline finished with success, oldest finish first."""
    found = []
    for record_path in runs_dir.glob("ssa-*/run_record.json"):
        record = reexecute_run.read_json(record_path) or {}
        if record.get("run_kind") == "study" and record.get("pipeline_status") == "success" and (record_path.parent / "out" / "results.json").exists():
            found.append((record.get("finished") or "", record_path.parent.name))
    return sorted(found)


def score(run_id: str, system: str = "SSA", seed: int = SEED) -> int:
    return int(hashlib.sha256(f"{seed}|{system}|{run_id}".encode()).hexdigest(), 16)


def second_execution_set(run_ids: list[str], system: str = "SSA", seed: int = SEED) -> list[str]:
    n = len(run_ids)
    k = min(n, max(FLOOR, math.ceil(FRACTION * n)))
    return sorted(sorted(run_ids, key=lambda r: score(r, system, seed))[:k], key=lambda r: score(r, system, seed))


def plan(runs_dir: Path, result_root: Path) -> list[tuple[str, int, bool]]:
    """[(run id, execution number, already done)] in the order they will run."""
    ordered = [run_id for _, run_id in counted_runs(runs_dir)]
    done = lambda run_id, n: (result_root / run_id / f"exec{n}" / "result.json").exists()  # noqa: E731
    steps = [(run_id, 1, done(run_id, 1)) for run_id in ordered]
    steps += [(run_id, 2, done(run_id, 2)) for run_id in ordered if run_id in set(second_execution_set(ordered))]
    return steps


def slot_busy(cpuset: str) -> list[str]:
    """SSA study containers that are running now (they hold CPUs 10-18)."""
    names = ssa_run.docker("ps", "--format", "{{.Names}}", check=False).stdout.split()
    return [n for n in names if n.startswith("ssa-ssa-") and cpuset == ssa_run.CPUSET]


def write_status(path: Path, **fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**fields, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))


def main(argv: list[str] | None = None, execute=reexecute_run.main, sleep=time.sleep, busy=slot_busy) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=ssa_run.RUNS_DIR)
    parser.add_argument("--result-root", type=Path, default=reexecute_run.RESULT_ROOT)
    parser.add_argument("--tool-image", required=True)
    parser.add_argument("--cpuset", default=ssa_run.CPUSET)
    parser.add_argument("--memory", default=ssa_run.MEMORY)
    parser.add_argument("--machine", default=None)
    parser.add_argument("--max-executions", type=int, default=0, help="stop after this many executions (0: all)")
    parser.add_argument("--stop-file", type=Path, default=None, help="default: <result-root>/REEXEC_STOP")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    stop_file = args.stop_file or args.result_root / "REEXEC_STOP"
    steps = plan(args.runs_dir, args.result_root)
    todo = [(r, n) for r, n, done in steps if not done]
    print(f"{len(steps)} executions planned ({sum(1 for _, n, _ in steps if n == 1)} first, {sum(1 for _, n, _ in steps if n == 2)} second); "
          f"{len(todo)} still to do")
    if args.dry_run:
        for run_id, n in todo:
            print(f"  would run {run_id} execution {n}")
        return 0
    status_path = args.result_root / "batch_status.json"
    finished_ok = failed = 0
    for index, (run_id, n) in enumerate(todo):
        if args.max_executions and index >= args.max_executions:
            break
        if stop_file.exists():
            write_status(status_path, state=f"Stopped: {stop_file} exists", done=index, remaining=len(todo) - index)
            print(f"Stopped: {stop_file} exists")
            return 0
        while busy(args.cpuset):
            write_status(status_path, state="Waiting: an SSA study run holds these CPUs", next=f"{run_id} execution {n}", done=index, remaining=len(todo) - index)
            sleep(60)
            if stop_file.exists():
                break
        if stop_file.exists():
            continue
        write_status(status_path, state=f"Running {run_id} execution {n}", done=index, remaining=len(todo) - index)
        command = ["--run-dir", str(args.runs_dir / run_id), "--execution", str(n), "--tool-image", args.tool_image,
                   "--result-root", str(args.result_root), "--cpuset", args.cpuset, "--memory", args.memory]
        if args.machine:
            command += ["--machine", args.machine]
        code = execute(command)
        if code == 0:
            finished_ok += 1
        else:
            failed += 1
    write_status(status_path, state="Done (this pass)", executed_ok=finished_ok, first_attempt_failed_or_not_started=failed)
    print(f"Done: {finished_ok} executions ran, {failed} failed or did not start")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
