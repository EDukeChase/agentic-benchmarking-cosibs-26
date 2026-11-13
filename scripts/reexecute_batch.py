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
import concurrent.futures
import hashlib
import json
import math
import queue
import subprocess
import sys
import threading
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


def parse_cpus(text: str) -> set[int]:
    """'10-18' or '9,19' or '10-11,14' as a set of CPU numbers."""
    cpus: set[int] = set()
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-")
            cpus |= set(range(int(low), int(high) + 1))
        elif part.strip():
            cpus.add(int(part))
    return cpus


def split_cpus(cpus: str, per_worker: int, workers: int) -> list[str]:
    """Disjoint CPU sets for the workers, in order: split_cpus('10-17', 2, 4) -> ['10,11', '12,13', '14,15', '16,17']."""
    available = sorted(parse_cpus(cpus))
    if per_worker * workers > len(available):
        raise ValueError(f"{workers} workers x {per_worker} CPUs need {per_worker * workers} CPUs, only {len(available)} given")
    return [",".join(str(c) for c in available[i * per_worker:(i + 1) * per_worker]) for i in range(workers)]


def slot_busy(cpuset: str) -> list[str]:
    """SSA study containers that are running now, if the CPUs asked for overlap the study slot (CPUs 10-18)."""
    if not (parse_cpus(cpuset) & parse_cpus(ssa_run.CPUSET)):
        return []
    names = ssa_run.docker("ps", "--format", "{{.Names}}", check=False).stdout.split()
    return [n for n in names if n.startswith("ssa-ssa-")]


def write_status(path: Path, **fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**fields, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))


def main(argv: list[str] | None = None, execute=reexecute_run.main, sleep=time.sleep, busy=slot_busy) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=ssa_run.RUNS_DIR)
    parser.add_argument("--result-root", type=Path, default=reexecute_run.RESULT_ROOT)
    parser.add_argument("--tool-image", required=True)
    parser.add_argument("--cpuset", default=ssa_run.CPUSET, help="CPUs to use; with --workers, the pool that is split between the workers")
    parser.add_argument("--memory", default=ssa_run.MEMORY, help="memory per execution")
    parser.add_argument("--workers", type=int, default=1, help="executions at the same time (each gets its own CPUs and memory)")
    parser.add_argument("--cpus-per-worker", type=int, default=2)
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
    try:
        slices = split_cpus(args.cpuset, args.cpus_per_worker, args.workers) if args.workers > 1 else [args.cpuset]
    except ValueError as error:
        print(f"Not started: {error}", file=sys.stderr)
        return 3
    if args.max_executions:
        todo = todo[: args.max_executions]
    free_slices: queue.Queue = queue.Queue()
    for item in slices:
        free_slices.put(item)
    counts = {"ok": 0, "failed": 0, "skipped": 0, "done": 0}
    lock = threading.Lock()

    def one(step):
        run_id, n = step
        if stop_file.exists():
            with lock:
                counts["skipped"] += 1
            return
        cpuset = free_slices.get()
        try:
            while busy(cpuset) and not stop_file.exists():
                write_status(status_path, state="Waiting: an SSA study run holds these CPUs", next=f"{run_id} execution {n}", **counts)
                sleep(60)
            if stop_file.exists():
                with lock:
                    counts["skipped"] += 1
                return
            with lock:
                write_status(status_path, state=f"Running (up to {len(slices)} at a time), next: {run_id} execution {n}", **counts)
            code = execute(command_for(run_id, n, cpuset, args.memory))
            with lock:
                counts["ok" if code == 0 else "failed"] += 1
                counts["done"] += 1
        finally:
            free_slices.put(cpuset)

    def command_for(run_id, n, cpuset, memory):
        command = ["--run-dir", str(args.runs_dir / run_id), "--execution", str(n), "--tool-image", args.tool_image,
                   "--result-root", str(args.result_root), "--cpuset", cpuset, "--memory", memory]
        return command + (["--machine", args.machine] if args.machine else [])

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slices)) as pool:
        list(pool.map(one, todo))

    # An execution killed for memory under a smaller limit than the study's is a resource artefact, not a result:
    # keep its folder under another name and run it again once at the study's memory, alone.
    redo = []
    if args.memory != ssa_run.MEMORY:
        for run_id, n in todo:
            folder = args.result_root / run_id / f"exec{n}"
            result = reexecute_run.read_json(folder / "result.json") or {}
            if result.get("oom_killed"):
                folder.rename(folder.with_name(f"exec{n}_killed_at_{args.memory}"))
                redo.append((run_id, n))
    for run_id, n in redo:
        if stop_file.exists():
            break
        write_status(status_path, state=f"Re-running {run_id} execution {n} at {ssa_run.MEMORY} (killed at {args.memory})", **counts)
        code = execute(command_for(run_id, n, args.cpuset, ssa_run.MEMORY))
        counts["ok" if code == 0 else "failed"] += 1
    write_status(status_path, state="Stopped by the stop file" if stop_file.exists() else "Done (this pass)",
                 executed_ok=counts["ok"], first_attempt_failed_or_not_started=counts["failed"], skipped=counts["skipped"], redone_at_full_memory=len(redo))
    print(f"Done: {counts['ok']} executions ran, {counts['failed']} failed or did not start, {counts['skipped']} skipped, {len(redo)} redone at full memory")
    return 0 if not counts["failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
