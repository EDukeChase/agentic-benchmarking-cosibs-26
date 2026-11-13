"""Re-execute a finished SSA run's own `reproduce.py` in a fresh container and compare it with what the agents reported.

    python3 scripts/reexecute_run.py --run-dir ~/ssa-runs/<run id> --tool-image ssa-study:<tag>
    python3 scripts/reexecute_run.py --run-dir <run> --execution 2 --tool-image ssa-study:<tag>     # a second execution
    python3 scripts/reexecute_run.py --run-dir <run> --tool-image ssa-study:<tag> --dry-run     # show the plan, run nothing

The run's code is used unchanged. This starts no model and costs nothing. For a finished run it:
1. copies the run's whole `out/` folder (the original run folder is only read) into a work folder, so the code finds what it
   expects next to it (for example packages the agents installed into out/vendor); the new outputs go to a separate folder,
2. checks the run's recorded package list against the image used for the re-execution,
3. runs `python <copy of out>/code/reproduce.py --data-dir <data> --out-dir <fresh folder>` in a fresh container: no network,
   unprivileged user, same CPUs and memory as a study run, data read-only, a time limit,
4. compares the new `results.json` / `predictions.csv` with the reported ones (src/evaluation/reexec.py, in the
   image; the tolerances are in scripts/reexec_tolerances.json),
5. writes `result.json`, the container log and the
   sampled memory into <result-root>/<run id>/exec<n>/, and one line to <result-root>/reexec_log.csv.
It needs the host's Docker only; paths, image names and the machine name are settings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import ssa_run  # noqa: E402  (data mount, docker helper, memory sampler, thread settings)
from src.evaluation.reexec_helpers import guess_failure_cause, package_differences, refused_output_folder  # noqa: E402  (no pandas: the host has none)

RESULT_ROOT = Path.home() / "ssa-reexec"
TOLERANCES = Path(__file__).with_name("reexec_tolerances.json")
AGENT_UID = "2000:2000"
LOG_HEADER = ["time", "run_id", "execution", "machine", "first_attempt", "repaired", "wall_min", "methods_within_tolerance",
              "methods_judged", "all_within_tolerance", "environment_image", "note"]


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def container_command(*, name: str, image: str, src_dir: Path, out_dir: Path | None, data_dir: Path, cpuset: str, memory: str,
                      hashseed: str | None = "5960") -> list[str]:
    """The `docker run` for the re-execution. No network, no GPU, unprivileged, data read-only."""
    command = ["docker", "run", "--name", name, "--runtime", "runc", "--network", "none",
               "--cpuset-cpus", cpuset, "--memory", memory, "--memory-swap", memory,
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", AGENT_UID,
               "--mount", f"type=bind,source={data_dir},target=/home/llmrel/task/data,readonly",
               "--mount", f"type=bind,source={src_dir},target=/work/src",
               *(["--mount", f"type=bind,source={out_dir},target=/work/out"] if out_dir else []),
               "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1", *(["-e", f"PYTHONHASHSEED={hashseed}"] if hashseed else [])]
    for key, value in ssa_run.THREADS.items():
        command += ["-e", f"{key}={value}"]
    return command + ["--workdir", "/work/src/code", "--entrypoint", "python", image,
                      "/work/src/code/reproduce.py", "--data-dir", "/home/llmrel/task/data",
                      "--out-dir", "/work/out" if out_dir else "/work/src/reexec_out"]


def reclaim(folder: Path, image: str) -> None:
    """Hand a folder the container's user wrote in back to the host user (files made as uid 2000 cannot be deleted otherwise)."""
    if folder.exists():
        subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "chown", "--mount", f"type=bind,source={folder},target=/w",
                        image, "-R", f"{os.getuid()}:{os.getgid()}", "/w"], capture_output=True)


def image_exists(tag: str) -> bool:
    return ssa_run.docker("image", "inspect", "--format", "{{.Id}}", tag, check=False).returncode == 0


def choose_images(record: dict, requested: str | None, tool_image: str) -> tuple[str, str | None]:
    """(environment image, note). The run's own image where it still exists; otherwise the tool image, said in the note."""
    if requested:
        return requested, "environment image given on the command line"
    own = record.get("image")
    if own and image_exists(own):
        current = ssa_run.docker("image", "inspect", "--format", "{{.Id}}", own, check=False).stdout.strip()
        if record.get("image_id") and current != record["image_id"]:
            return own, f"the tag {own} now points to a different image than the run used (see package_differences)"
        return own, None
    return tool_image, f"the run's image {own} is not on this machine; the tool image was used (see package_differences)"


def installed_packages(image: str) -> list[str]:
    out = ssa_run.docker("run", "--rm", "--network", "none", "--entrypoint", "pip", image, "freeze", check=False)
    return out.stdout.splitlines() if out.returncode == 0 else []


def run_container(command: list[str], name: str, log_path: Path, max_seconds: int, sample_path: Path) -> dict:
    sampler = ssa_run.MemorySampler(name, sample_path)
    started = time.monotonic()
    timed_out = False
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        sampler.start()
        try:
            process.wait(timeout=max_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            ssa_run.docker("stop", "-t", "30", name, check=False)
            process.wait()
    memory = sampler.finish()
    wall = time.monotonic() - started
    exit_code = int(ssa_run.docker("inspect", "--format", "{{.State.ExitCode}}", name, check=False).stdout.strip() or -1)
    oom = ssa_run.docker("inspect", "--format", "{{.State.OOMKilled}}", name, check=False).stdout.strip() == "true"
    ssa_run.docker("rm", "-f", name, check=False)
    return {"exit_code": exit_code, "timed_out": timed_out, "oom_killed": oom, "wall_seconds": wall, "memory": memory}


def compare_in_image(tool_image: str, reported_out: Path, new_out: Path, result_dir: Path, tolerances_file: Path) -> dict | None:
    """The numbers comparison runs in the image (it needs pandas), read-only on both folders."""
    command = ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python",
               "--mount", f"type=bind,source={reported_out},target=/cmp/reported,readonly",
               "--mount", f"type=bind,source={new_out},target=/cmp/new,readonly",
               "--mount", f"type=bind,source={result_dir},target=/cmp/result",
               "--mount", f"type=bind,source={tolerances_file},target=/cmp/tolerances.json,readonly",
               tool_image, "scripts/reexec_compare.py", "--reported-dir", "/cmp/reported", "--new-dir", "/cmp/new",
               "--out", "/cmp/result/compare.json", "--tolerances", "/cmp/tolerances.json"]
    done = subprocess.run(command, capture_output=True, text=True)
    (result_dir / "compare.log").write_text(done.stdout + done.stderr)
    return read_json(result_dir / "compare.json")


def assemble(*, run_id: str, execution: int, record: dict, machine: str, started: str, finished: str, ran: dict, outputs_found: bool,
             compare: dict | None, environment_image: str, image_note: str | None, packages: dict, cpuset: str, memory: str,
             tolerances_sha: str | None, log_text: str, repaired: bool = False, out_inside_copy: bool = False, hashseed_fixed: bool = False) -> dict:
    first_attempt = "ran" if (ran["exit_code"] == 0 and not ran["timed_out"] and outputs_found) else "failed"
    methods = (compare or {}).get("methods") or []
    judged = [m for m in methods if m["within_tolerance"] is not None]
    cause = None
    if first_attempt == "failed":
        cause = "time limit" if ran["timed_out"] else "out of memory" if ran["oom_killed"] else guess_failure_cause(log_text)
    return {
        "run_id": run_id, "n": execution, "kind": "scripted", "machine": machine, "started": started, "finished": finished,
        "first_attempt": first_attempt, "repaired": repaired, "exit_code": ran["exit_code"], "timed_out": ran["timed_out"],
        "oom_killed": ran["oom_killed"], "failure_cause_guess": cause,
        "wall_min": round(ran["wall_seconds"] / 60, 1), "changelog_pointer": None,
        "taxonomy_run_level": "a" if first_attempt == "failed" else ("none" if judged and all(m["within_tolerance"] for m in judged) else "b"),
        "all_within_tolerance": bool(judged) and all(m["within_tolerance"] for m in judged) and len(judged) == len(methods),
        "methods": methods,
        "limits": {"cpuset": cpuset, "memory": memory, "network": "none", "user": AGENT_UID},
        "run_image": record.get("image"), "environment_image": environment_image, "image_note": image_note,
        "package_differences": packages, "tolerances_sha256": tolerances_sha,
        "peak_memory_sampled": {k: ran["memory"].get(k) for k in ("anon_peak_sampled_gb", "current_peak_sampled_gb", "memory_samples")},
        "out_dir_inside_copy": out_inside_copy, "pythonhashseed_fixed": "5960" if hashseed_fixed else None, "script_version": script_version(), "reported_run": {"started": record.get("started"), "image_git_commit": record.get("image_git_commit"), "prompt_sha256": record.get("prompt_sha256")},
    }


def script_version() -> dict:
    """Which version of this script made a result (commit of the repo it sits in, and whether it had uncommitted changes)."""
    repo = Path(__file__).resolve().parent.parent
    def git(*a):
        done = subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain"))}


def fallback_argv(args, subdir: str) -> list[str]:
    """The same re-execution again, with the new outputs inside the copy of the run's out folder (the standard B fallback)."""
    argv = ["--run-dir", str(args.run_dir), "--execution", str(args.execution), "--tool-image", args.tool_image,
            "--result-root", str(args.result_root), "--cpuset", args.cpuset, "--memory", args.memory, "--max-hours", str(args.max_hours),
            "--machine", args.machine, "--tolerances", str(args.tolerances), "--dataset", args.dataset,
            "--out-inside-copy", "--result-subdir", f"{subdir}_outinside", "--no-fallback"]
    if args.fix_hashseed:
        argv.append("--fix-hashseed")
    if args.data_dir:
        argv += ["--data-dir", str(args.data_dir)]
    if args.environment_image:
        argv += ["--environment-image", args.environment_image]
    if args.work_root:
        argv += ["--work-root", str(args.work_root)]
    return argv


def append_log(root: Path, result: dict, note: str = "", ran_override=None, within_override=None) -> None:
    path = root / "reexec_log.csv"
    new = not path.exists()
    judged = [m for m in result["methods"] if m["within_tolerance"] is not None]
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(LOG_HEADER)
        writer.writerow([datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"), result["run_id"], result["n"], result["machine"],
                         ran_override or result["first_attempt"], result["repaired"], result["wall_min"],
                         sum(1 for m in judged if m["within_tolerance"]), len(judged),
                         within_override if within_override is not None else result["all_within_tolerance"],
                         result["environment_image"], note])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path, help="the finished run's folder (out/ and records/ inside)")
    parser.add_argument("--execution", type=int, default=1, help="1, or 2 for the second execution of a subsample run")
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--work-root", type=Path, default=None, help="default: <result-root>/_work")
    parser.add_argument("--dataset", default=ssa_run.DATASET)
    parser.add_argument("--data-dir", type=Path, default=None, help="default: the mounted data image of the dataset")
    parser.add_argument("--environment-image", default=None, help="default: the image the run used, if it is still here")
    parser.add_argument("--tool-image", required=True, help="image with the comparison code (an ssa-study tag built from this repo's current commit)")
    parser.add_argument("--cpuset", default=ssa_run.CPUSET)
    parser.add_argument("--memory", default=ssa_run.MEMORY)
    parser.add_argument("--max-hours", type=float, default=4.0)
    parser.add_argument("--machine", default=socket.gethostname())
    parser.add_argument("--tolerances", type=Path, default=TOLERANCES)
    parser.add_argument("--code-dir", type=Path, default=None, help="run this code instead of the run's own out/code (a repaired copy)")
    parser.add_argument("--result-subdir", default=None, help="result folder name under <result-root>/<run id>/ (default exec<n>)")
    parser.add_argument("--repaired", action="store_true", help="label the result as the execution of repaired code")
    parser.add_argument("--out-inside-copy", action="store_true", help="diagnostic: write the new outputs to a folder INSIDE the copy of out/ "
                        "(some agent code refuses an output folder outside its own out folder)")
    parser.add_argument("--fix-hashseed", action="store_true", help="fix PYTHONHASHSEED at 5960 (the official SSA batch did; the original runs did not, and a test of 5 runs showed no difference)")
    parser.add_argument("--no-fallback", action="store_true", help="do not re-run with the outputs inside the copy of out/ when the code refuses the separate output folder")
    parser.add_argument("--keep-work", action="store_true", help="keep the work folder (copy of out/ and the new outputs) after the run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    run_id = run_dir.name
    record = read_json(run_dir / "run_record.json") or {}
    out_dir, reported_results = run_dir / "out", run_dir / "out" / "results.json"
    problems = [f"{p} is missing" for p in (out_dir / "code" / "reproduce.py", reported_results, out_dir / "predictions.csv") if not p.exists()]
    if record.get("pipeline_status") != "success":
        problems.append(f"the run's pipeline status is {record.get('pipeline_status')!r}, not success")
    if problems:
        print("Not started: " + "; ".join(problems), file=sys.stderr)
        return 3
    data_dir = args.data_dir or ssa_run.data_mount_for(ssa_run.image_meta(args.dataset))
    environment_image, image_note = choose_images(record, args.environment_image, args.tool_image)
    subdir = args.result_subdir or f"exec{args.execution}"
    result_dir = args.result_root / run_id / subdir
    work = (args.work_root or args.result_root / "_work") / run_id / subdir
    name = f"ssa-reexec-{run_id[-4:]}-{subdir.replace('/', '-')}"
    print(f"{run_id} execution {args.execution}: environment image {environment_image}; data {data_dir}; CPUs {args.cpuset}, {args.memory}; "
          f"results to {result_dir}" + (f" ({image_note})" if image_note else ""))
    if args.dry_run:
        return 0
    if (result_dir / "result.json").exists():
        print(f"Not started: {result_dir} already has a result (move it away to repeat)", file=sys.stderr)
        return 3

    reclaim(work, args.tool_image)
    shutil.rmtree(work, ignore_errors=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    src_dir, new_out = work / "src", work / "out"
    skip_symlinks = lambda folder, names: [n for n in names if (Path(folder) / n).is_symlink()]  # noqa: E731
    shutil.copytree(out_dir, src_dir, ignore=skip_symlinks)
    if args.code_dir:  # a repaired copy of the code replaces the run's own code folder
        shutil.rmtree(src_dir / "code")
        shutil.copytree(args.code_dir, src_dir / "code", ignore=skip_symlinks)
    if args.out_inside_copy:
        new_out = src_dir / "reexec_out"
    new_out.mkdir(parents=True, exist_ok=True)
    os.chmod(work, 0o777)
    os.chmod(new_out, 0o777)
    for root, dirs, files in os.walk(src_dir):  # the container's unprivileged user must be able to read and write in the copy
        os.chmod(root, 0o777)
        for item in files:
            os.chmod(Path(root) / item, 0o666)

    recorded = (run_dir / "records" / "packages.txt").read_text().splitlines() if (run_dir / "records" / "packages.txt").exists() else []
    packages = package_differences(recorded, installed_packages(environment_image))
    started = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    command = container_command(name=name, image=environment_image, src_dir=src_dir, out_dir=None if args.out_inside_copy else new_out, data_dir=data_dir,
                                cpuset=args.cpuset, memory=args.memory, hashseed="5960" if args.fix_hashseed else None)
    ran = run_container(command, name, result_dir / "reexec.log", int(args.max_hours * 3600), result_dir / ssa_run.SAMPLES_NAME)
    finished = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    reclaim(work, args.tool_image)
    outputs_found = (new_out / "results.json").exists() and (new_out / "predictions.csv").exists()
    for item in ("results.json", "predictions.csv"):
        if (new_out / item).exists():
            shutil.copy2(new_out / item, result_dir / f"reexecuted_{item}")
    compare = compare_in_image(args.tool_image, out_dir, new_out, result_dir, args.tolerances) if outputs_found else None
    try:
        log_text = (result_dir / "reexec.log").read_text(errors="replace")[-20000:]
    except OSError:
        log_text = ""
    result = assemble(run_id=run_id, execution=args.execution, record=record, machine=args.machine, started=started, finished=finished,
                      ran=ran, outputs_found=outputs_found, compare=compare, environment_image=environment_image, image_note=image_note,
                      packages=packages, cpuset=args.cpuset, memory=args.memory, tolerances_sha=sha256(args.tolerances), log_text=log_text,
                      repaired=args.repaired, out_inside_copy=args.out_inside_copy, hashseed_fixed=args.fix_hashseed)
    log_note = image_note or ""
    fallback_ran = False
    if (result["first_attempt"] == "failed" and refused_output_folder(log_text) and not (args.no_fallback or args.out_inside_copy or args.repaired)):
        # Standard B: the code refused the separate output folder; run it again with the outputs inside the copy of out/.
        # The strict failure stays on record in this result; the fallback has its own folder.
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)
        main(fallback_argv(args, subdir))
        fb = read_json(args.result_root / run_id / f"{subdir}_outinside" / "result.json") or {}
        fallback_ran = fb.get("first_attempt") == "ran"
        result["fallback_output_inside_out_folder"] = {
            "reason": "the code refused an output folder outside its own out folder (see reexec.log)", "result_folder": f"{subdir}_outinside",
            "first_attempt": fb.get("first_attempt"), "all_within_tolerance": fb.get("all_within_tolerance"), "wall_min": fb.get("wall_min")}
        log_note = "output folder had to be inside out/ (fallback)"
    result["first_attempt_standard_b"] = "ran" if (result["first_attempt"] == "ran" or fallback_ran) else "failed"
    (result_dir / "result.json").write_text(json.dumps(result, indent=1))
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    append_log(args.result_root, result, note=log_note, ran_override=("ran" if fallback_ran else None),
               within_override=(result["fallback_output_inside_out_folder"]["all_within_tolerance"] if fallback_ran else None))
    judged = [m for m in result["methods"] if m["within_tolerance"] is not None]
    print(f"{run_id} execution {args.execution}: first attempt {result['first_attempt']} in {result['wall_min']} min; "
          f"{sum(1 for m in judged if m['within_tolerance'])} of {len(judged)} methods within tolerance"
          + (f"; cause guess: {result['failure_cause_guess']}" if result["failure_cause_guess"] else ""))
    return 0 if (result["first_attempt"] == "ran" or fallback_ran) else 2


if __name__ == "__main__":
    raise SystemExit(main())
