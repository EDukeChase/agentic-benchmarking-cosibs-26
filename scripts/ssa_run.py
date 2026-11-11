"""Run the SSA pipeline in a container, as one study run or as a queue of runs.

Runs on the host (Ubuntu, with `docker`), standard library only. The container
cannot reach the data image from a VS Code dev container; it has to be started
from Ubuntu's docker. For every run this script:

  1. verifies the data image's file checksums, and the study prompt's SHA-256,
  2. checks the setup (VPN reachability, Azure and Google credentials, no GPU),
  3. starts the image with 9 pinned CPUs and 11 GB, the data mounted read-only,
     empty output and records folders, and a hard time limit,
  4. after the run: re-verifies the data checksums, runs the audit on the output,
     and writes one run record with the same kind of fields the AISS runs get.

Commands:
  login        one-time `az login` inside the image (the cache lives in ~/.ssa-azure)
  check        the setup checks of step 2 only
  run          one run
  queue --n N  N runs in a row, waiting and retrying when the infrastructure fails

A failure of the infrastructure (VPN down, expired login, API quota, a container killed from
outside) is not a study outcome. Such a run is marked counts_as_study_run=false and, in a queue,
replaced by a new run after a wait; a run is never continued. A failure that is neither an
infrastructure failure nor a recognised agent outcome (time limit, out of memory, a failed method,
an agent that gave up) is "harness_suspect": it is held, not counted, and two in a row stop the queue.

Example:
  python3 scripts/ssa_run.py run --prompt ~/cosibs-notes/pilot-01__v05.claude-science.md \\
      --prompt-sha256 <sha256 of that file>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATASET = "study_v1"
IMAGE = "ssa-study:dev"
IMAGE_META_DIR = Path("/mnt/d/wsl-images/data")
DATA_MOUNT_ROOT = Path("/mnt/wsl")
RUNS_DIR = Path.home() / "ssa-runs"
AZURE_DIR = Path.home() / ".ssa-azure"
ADC_FILE = Path("/mnt/c/Users/dukec/AppData/Roaming/gcloud/application_default_credentials.json")
# Where a finished study run is copied to (same places as the AISS runs; never visible to agent runs).
OUT_COPY_ROOT = Path("/mnt/c/Users/dukec/agent-runs")
RECORDS_COPY_ROOT = Path("/mnt/d/masters-project/run-records/ssa")
REPO = Path(__file__).resolve().parent.parent
# run_record.json "status": finished, stopped_early. A run that cannot start (data checksums wrong,
# VPN or login not ready, study settings not met) writes no record; the script prints why and exits 3.
AZURE_HOST = "bpsmar-ai-openai-1.openai.azure.com"
CPUSET = "10-18"
MEMORY = "11g"
THREADS = {"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
           "NUMEXPR_NUM_THREADS": "8", "LOKY_MAX_CPU_COUNT": "9"}
GRACE_SECONDS = 180  # after the study's time limit, for the pipeline to write its records

# Exception types that mean "the infrastructure failed", not "the agents failed".
INFRASTRUCTURE_ERROR_TYPES = (
    "APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError",
    "ServiceUnavailable", "ResourceExhausted", "ClientAuthenticationError",
    "CredentialUnavailableError", "ConnectError", "ConnectTimeout", "ServerError",
    "RefreshError", "TransportError", "KeyboardInterrupt",
)
INFRASTRUCTURE_MESSAGES = ("DefaultAzureCredential failed", "AzureCliCredential", "az login", "Please run 'az login'")
RATE_LIMIT_HINTS = ("RateLimitError", "ResourceExhausted", "429", "quota")
# Pipeline errors that are the agents' own doing. Anything else is held for review, not counted.
AGENT_OUTCOME_ERROR_TYPES = ("StageTimeoutError", "UnknownPatientIdsError", "GraphRecursionError",
                             "ValidationError", "OutputParserException")
AGENT_OUTCOME_MESSAGES = ("no model.py for",)
KILLED_FROM_OUTSIDE = (137, 143)


class SetupError(RuntimeError):
    """The environment is not ready. Not a study outcome."""


# ---------------------------------------------------------------- pure helpers

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sums(lines: list[str]) -> dict[str, str]:
    """`sha256sum` lines ("<hash>  <file>") to {file: hash}."""
    sums = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line.strip())
        if not match:
            raise SetupError(f"Cannot read checksum line: {line!r}")
        sums[match.group(2)] = match.group(1).lower()
    return sums


def verify_data(folder: Path, sums: dict[str, str], hasher=sha256_file) -> dict:
    present = {p.name for p in folder.iterdir() if p.is_file()}
    mismatched = sorted(name for name in sums if name in present and hasher(folder / name) != sums[name])
    result = {
        "files_expected": len(sums),
        "missing": sorted(set(sums) - present),
        "extra": sorted(present - set(sums)),
        "mismatched": mismatched,
    }
    result["ok"] = not (result["missing"] or result["extra"] or result["mismatched"])
    return result


def classify_failure(manifest: dict, error: dict | None) -> str | None:
    """"infrastructure", "rate_limit", or None. Looks at the pipeline's own errors only.

    This reads exception types and messages the pipeline recorded, not the agents'
    text, so an agent printing "429" in its own code does not count. It is a
    heuristic and may miss errors it does not know.
    """
    texts = []
    if error:
        texts.append(f"{error.get('error_type', '')}: {error.get('error', '')}")
    texts.extend(str(v) for v in (manifest.get("stage_failures") or {}).values())
    if not texts:
        return None
    joined = "\n".join(texts)
    infrastructure = any(t in joined for t in INFRASTRUCTURE_ERROR_TYPES + INFRASTRUCTURE_MESSAGES)
    if not infrastructure:
        return None
    return "rate_limit" if any(h in joined for h in RATE_LIMIT_HINTS) else "infrastructure"


def stage_failure_is_agent_outcome(text: str) -> bool:
    if any(text.startswith(m) for m in AGENT_OUTCOME_MESSAGES):
        return True
    return text.split(":", 1)[0].strip() in AGENT_OUTCOME_ERROR_TYPES


def decide_failure(manifest: dict | None, error: dict | None, exit_code: int, stopped_early: bool,
                   oom: bool = False) -> tuple[str | None, str | None]:
    """(failure_class, infrastructure_error_seen) for a finished run.

    failure_class is None (the run counts), "infrastructure", "rate_limit" or "harness_suspect".
    - Killed at the time limit, or out of memory: the agents' outcome, it counts.
    - An infrastructure error in any stage (even if the pipeline went on and "succeeded" without that
      stage's work): the stage never got a fair chance, so the run does not count.
    - A failed pipeline whose errors are all recognised agent outcomes counts. An unrecognised error
      (a KeyError in our own code, an HTTP 400 and so on) is held for review.
    - No records at all: killed from outside (docker or WSL restart) is infrastructure; any other
      early exit is held for review.
    """
    seen = classify_failure(manifest or {}, error)
    if stopped_early or oom:
        return None, seen
    if manifest is None and error is None:
        return ("infrastructure" if exit_code in KILLED_FROM_OUTSIDE else "harness_suspect"), seen
    if seen:
        return seen, seen
    texts = [f"{error['error_type']}: {error.get('error', '')}"] if error else []
    texts.extend(str(v) for v in ((manifest or {}).get("stage_failures") or {}).values())
    if all(stage_failure_is_agent_outcome(t) for t in texts):
        return None, None
    return "harness_suspect", None


def study_settings_problems(args, image_label: dict, repo_commit: str | None, repo_dirty: bool | None) -> list[str]:
    """Why this run may not be a study run. An empty list means it may be.

    A study run uses the study settings and an image built from the committed code, so the
    record can say exactly what produced it.
    """
    problems = []
    if args.dataset != DATASET:
        problems.append(f"dataset is {args.dataset}, a study run uses {DATASET}")
    if args.fixture_literature:
        problems.append("the literature fixture is on (smoke tests only)")
    if float(args.max_hours) != 4.0:
        problems.append(f"time limit is {args.max_hours} h, the study uses 4")
    if args.memory != MEMORY:
        problems.append(f"memory is {args.memory}, the study uses {MEMORY}")
    if args.cpuset not in ("0-8", CPUSET):
        problems.append(f"CPU range is {args.cpuset}, the study uses 0-8 or {CPUSET}")
    if not args.prompt_sha256:
        problems.append("--prompt-sha256 is not given")
    commit = image_label.get("git_commit")
    if not commit or commit == "unknown":
        problems.append("the image has no git_commit label (rebuild it with --build-arg GIT_COMMIT=...)")
    elif image_label.get("git_dirty") != "false":
        problems.append("the image was built from a working tree with uncommitted changes")
    elif repo_commit and commit != repo_commit:
        problems.append(f"the image was built from {commit[:10]} but the repo is at {repo_commit[:10]}")
    if repo_dirty:
        problems.append("the repo has uncommitted changes")
    return problems


def repo_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


# ---------------------------------------------------------------- memory samples

CGROUP_DOCKER = Path("/sys/fs/cgroup/docker")
SAMPLE_SECONDS = 5
SAMPLES_NAME = "memory_samples.csv"


def read_memory_stat(cgroup: Path) -> dict | None:
    """anon (the processes' own memory), file (cache) and current, in bytes, from a container's cgroup."""
    try:
        stat = {}
        for line in (cgroup / "memory.stat").read_text().splitlines():
            key, _, value = line.partition(" ")
            if key in ("anon", "file"):
                stat[key] = int(value)
        stat["current"] = int((cgroup / "memory.current").read_text())
        return stat if "anon" in stat and "file" in stat else None
    except (OSError, ValueError):
        return None


class MemorySampler(threading.Thread):
    """Reads the container's anonymous and cache memory every few seconds, from the host.

    The container's own peak (memory.peak) includes the file cache and is lost when the container is
    killed. The samples are written as they are taken, so a killed run keeps them, and the largest
    anonymous value goes into the run record. It only reads; it never changes the container. If the
    cgroup cannot be found or read, the record says so (None) and the run is not affected.
    """

    def __init__(self, container: str, log_path: Path, root: Path = CGROUP_DOCKER, interval: float = SAMPLE_SECONDS):
        super().__init__(daemon=True)
        self.container, self.log_path, self.root, self.interval = container, log_path, root, interval
        self.stopper = threading.Event()
        self.cgroup: Path | None = None
        self.samples = 0
        self.peak_anon = 0
        self.peak_anon_time: str | None = None
        self.peak_current = 0

    def _find_cgroup(self) -> None:
        container_id = docker("inspect", "--format", "{{.Id}}", self.container, check=False).stdout.strip()
        if container_id and (self.root / container_id).is_dir():
            self.cgroup = self.root / container_id

    def sample_once(self, handle) -> None:
        if self.cgroup is None:
            self._find_cgroup()
        stat = read_memory_stat(self.cgroup) if self.cgroup else None
        if stat is None:
            return
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        self.samples += 1
        self.peak_current = max(self.peak_current, stat["current"])
        if stat["anon"] > self.peak_anon:
            self.peak_anon, self.peak_anon_time = stat["anon"], stamp
        handle.write(f"{stamp},{stat['anon']},{stat['file']},{stat['current']}\n")
        handle.flush()

    def run(self) -> None:
        try:
            with self.log_path.open("w") as handle:
                handle.write("time,anon_bytes,file_bytes,current_bytes\n")
                while not self.stopper.is_set():
                    try:
                        self.sample_once(handle)
                    except Exception:  # sampling must never disturb a run
                        pass
                    self.stopper.wait(self.interval)
        except OSError:
            pass

    def finish(self) -> dict:
        self.stopper.set()
        if self.is_alive():
            self.join(timeout=10)
        gb = lambda value: round(value / 1024**3, 2) if self.samples else None  # noqa: E731
        return {
            "anon_peak_sampled_gb": gb(self.peak_anon), "anon_peak_time": self.peak_anon_time,
            "current_peak_sampled_gb": gb(self.peak_current), "memory_samples": self.samples,
            "memory_sample_seconds": self.interval, "memory_samples_file": SAMPLES_NAME,
        }


def image_labels(image: str) -> dict:
    out = docker("image", "inspect", "--format", "{{json .Config.Labels}}", image, check=False).stdout.strip()
    try:
        return json.loads(out) or {}
    except json.JSONDecodeError:
        return {}


def container_args(*, name: str, image: str, run_dir: Path, data_mount: Path, prompt: Path, prompt_sha256: str,
                   run_id: str, max_seconds: int, aoai_ip: str, fixture_literature: bool = False,
                   cpuset: str = CPUSET, memory: str = MEMORY, adc_file: Path = ADC_FILE,
                   azure_dir: Path = AZURE_DIR, command: list[str] | None = None,
                   entrypoint: str | None = None, interactive: bool = False) -> list[str]:
    """The `docker run` command line. Never asks for a GPU."""
    args = ["docker", "run", "--name", name, "--runtime", "runc",
            "--cpuset-cpus", cpuset, "--memory", memory, "--memory-swap", memory,
            "--cap-drop", "ALL", "--cap-add", "SETUID", "--cap-add", "SETGID", "--cap-add", "DAC_OVERRIDE",
            "--cap-add", "CHOWN", "--security-opt", "no-new-privileges",
            "--add-host", f"{AZURE_HOST}:{aoai_ip}",
            "--mount", f"type=bind,source={data_mount},target=/home/llmrel/task/data,readonly",
            "--mount", f"type=bind,source={run_dir / 'out'},target=/home/llmrel/task/out",
            "--mount", f"type=bind,source={run_dir / 'records'},target=/srv/ssa/records",
            "--mount", f"type=bind,source={prompt},target=/srv/ssa/study_prompt.md,readonly",
            "--mount", f"type=bind,source={adc_file},target=/srv/ssa/google-adc.json,readonly",
            "--mount", f"type=bind,source={azure_dir},target=/root/.azure",
            "-e", "GOOGLE_APPLICATION_CREDENTIALS=/srv/ssa/google-adc.json",
            "-e", "GOOGLE_CLOUD_PROJECT=gac-som-dbmi-bpsmar-app-59",
            "-e", f"STUDY_PROMPT_SHA256={prompt_sha256}",
            "-e", f"BENCHMARK_RUN_ID={run_id}",
            "-e", f"STUDY_DEADLINE_SECONDS={max_seconds}"]
    for key, value in THREADS.items():
        args += ["-e", f"{key}={value}"]
    if fixture_literature:
        args += ["-e", "USE_LITERATURE_FIXTURE=1"]
    if interactive:
        args += ["-it"]
    if entrypoint:
        args += ["--entrypoint", entrypoint]
    args.append(image)
    return args + (command or [])


def now() -> datetime:
    return datetime.now(timezone.utc)


def local_stamp(moment: datetime | None = None) -> str:
    """Local time without a zone, to the second: the format of the AISS run records."""
    return (moment or datetime.now()).astimezone().replace(tzinfo=None).isoformat(timespec="seconds")


def verified_text(result: dict, mounted_read_only: bool | None) -> str:
    """The same wording the AISS records use for data_verified_at_start."""
    total = result["files_expected"]
    bad = len(result["mismatched"]) + len(result["missing"])
    files = total - len(result["missing"]) + len(result["extra"])
    mount = {True: "mounted read-only", False: "NOT mounted read-only", None: "mount options unknown"}[mounted_read_only]
    return f"{total - bad}/{total} checksums OK, {files} files in folder (expected {total}); {mount}"


def mounted_read_only(path: Path) -> bool | None:
    try:
        options = subprocess.run(["findmnt", "-n", "-o", "OPTIONS", "--target", str(path)],
                                 capture_output=True, text=True, check=True).stdout.strip().split(",")
    except (OSError, subprocess.CalledProcessError):
        return None
    return "ro" in options


def version_of(prompt: Path) -> str | None:
    match = re.search(r"__(v\d+)", prompt.name)
    return match.group(1) if match else None


# ---------------------------------------------------------------- docker and host helpers

def docker(*args: str, check: bool = True, timeout: float | None = None, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=check, timeout=timeout, text=True, capture_output=capture)


def image_meta(dataset: str) -> dict:
    path = IMAGE_META_DIR / f"{dataset}.json"
    if not path.exists():
        raise SetupError(f"No image metadata at {path}; the data image for {dataset} is not built yet.")
    return json.loads(path.read_text())


def data_mount_for(meta: dict) -> Path:
    mount = DATA_MOUNT_ROOT / meta["mount_name"]
    if not mount.is_dir():
        raise SetupError(f"Data image not mounted at {mount}.")
    return mount


def aoai_ip() -> str:
    ip = os.getenv("AOAI_PRIVATE_IP")
    if not ip:
        raise SetupError("AOAI_PRIVATE_IP is not set in this shell.")
    return ip


def check_setup(args, meta: dict, quiet: bool = False) -> None:
    """Everything that must be true before a run starts. Raises SetupError."""
    problems = []
    runtime = docker("info", "--format", "{{.DefaultRuntime}}").stdout.strip()
    if runtime != "runc":
        problems.append(f"Docker's default runtime is {runtime!r}, not 'runc' (GPU risk).")
    if docker("image", "inspect", args.image, check=False).returncode != 0:
        problems.append(f"Image {args.image} is not built (docker build -t {args.image} .).")
    if not ADC_FILE.exists():
        problems.append(f"Google credentials not found at {ADC_FILE}.")
    if not AZURE_DIR.exists():
        problems.append(f"No Azure login cache at {AZURE_DIR}; run: python3 scripts/ssa_run.py login")
    try:
        with socket.create_connection((aoai_ip(), 443), timeout=5):
            pass
    except OSError as error:
        problems.append(f"Cannot reach the Azure private endpoint ({error}). Is the campus VPN connected?")
    except SetupError as error:
        problems.append(str(error))
    if problems:
        raise SetupError("\n".join(problems))
    # Credentials actually work (token fetch only; nothing is printed except ok/failed).
    probe = (
        "import subprocess,sys\n"
        "r=subprocess.run(['az','account','get-access-token','--resource','https://cognitiveservices.azure.com','-o','none'],capture_output=True,text=True)\n"
        "print('azure','ok' if r.returncode==0 else 'FAILED '+r.stderr.strip()[-200:])\n"
        "import google.auth,google.auth.transport.requests as t\n"
        "try:\n"
        "    c,_=google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform']);c.refresh(t.Request());print('google ok')\n"
        "except Exception as e:\n"
        "    print('google FAILED',type(e).__name__)\n"
    )
    name = f"ssa-check-{secrets.token_hex(3)}"
    scratch = Path(args.runs_dir) / ".check"
    (scratch / "out").mkdir(parents=True, exist_ok=True)
    (scratch / "records").mkdir(exist_ok=True)
    for sub in ("out", "records"):
        os.chmod(scratch / sub, 0o777)
    command = container_args(
        name=name, image=args.image, run_dir=scratch, data_mount=data_mount_for(meta), prompt=Path(args.prompt),
        prompt_sha256="x", run_id="check", max_seconds=60, aoai_ip=aoai_ip(), entrypoint="python",
        command=["-c", probe],
    )
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    output = result.stdout.strip()
    if "azure ok" not in output or "google ok" not in output:
        raise SetupError(f"Credential check failed:\n{output}\n{result.stderr.strip()[-300:]}")
    if not quiet:
        print("Setup OK: no GPU runtime, image present, VPN reachable, Azure and Google credentials work.")


# ---------------------------------------------------------------- one run

def run_once(args, meta: dict, attempt: int = 1) -> dict:
    """One run. Returns the run record. Raises SetupError if the setup is not ready."""
    prompt = Path(args.prompt).resolve()
    prompt_sha = sha256_file(prompt)
    if args.prompt_sha256 and prompt_sha != args.prompt_sha256.lower():
        raise SetupError(f"Prompt file has SHA-256 {prompt_sha}, expected {args.prompt_sha256}.")
    args.prompt = str(prompt)
    labels = image_labels(args.image)
    repo_commit, repo_dirty = repo_state()
    if args.kind == "study":
        problems = study_settings_problems(args, labels, repo_commit, repo_dirty)
        if problems:
            raise SetupError("Not a study run:\n  - " + "\n  - ".join(problems))
    sums = parse_sums(meta["expected_sums"])
    data_mount = data_mount_for(meta)
    before = verify_data(data_mount, sums)
    if not before["ok"]:
        raise SetupError(f"Data checksums do not match before the run: {before}")
    check_setup(args, meta, quiet=True)

    run_id = f"ssa-{now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
    run_dir = Path(args.runs_dir) / run_id
    for sub in ("out", "records"):
        (run_dir / sub).mkdir(parents=True)
        os.chmod(run_dir / sub, 0o777)  # the container's root has no host privileges
    max_seconds = int(args.max_hours * 3600)
    started = now()
    read_only = mounted_read_only(data_mount)
    record = {
        # names follow the AISS registry entries (wsl_runs.json) where there is a counterpart
        "run_id": run_id, "system": "SSA", "run_kind": args.kind, "attempt": attempt, "dataset": meta["dataset"],
        "created": local_stamp(), "started": local_stamp(),
        "deadline": local_stamp(started + timedelta(seconds=max_seconds)), "max_hours": args.max_hours,
        "limits_requested": {"cpus": args.cpuset, "mem_gb": float(args.memory.rstrip("g")), "swap": "none", "gpu": False},
        "threads_env": THREADS,
        "literature_setting": "fixture" if args.fixture_literature else "live",
        "data_mode": "image", "data_image_sha256": meta.get("sha256"), "data_image_built": meta.get("built"),
        "data_verified_at_start": verified_text(before, read_only),
        "prompt_file": str(prompt), "prompt_sha256": prompt_sha, "prompt_version": version_of(prompt),
        "image": args.image, "image_id": docker("image", "inspect", "--format", "{{.Id}}", args.image).stdout.strip(),
        "output_dir": str(run_dir / "out"), "records_dir": str(run_dir / "records"),
        "network": "unrestricted",
        "image_git_commit": labels.get("git_commit"), "image_git_dirty": labels.get("git_dirty"),
        "repo_commit_at_start": repo_commit, "repo_dirty_at_start": repo_dirty,
    }
    name = f"ssa-{run_id}"
    command = container_args(
        name=name, image=args.image, run_dir=run_dir, data_mount=data_mount, prompt=prompt, prompt_sha256=prompt_sha,
        run_id=run_id, max_seconds=max_seconds, aoai_ip=aoai_ip(), fixture_literature=args.fixture_literature,
        cpuset=args.cpuset, memory=args.memory,
    )
    print(f"{run_id}: starting (limit {args.max_hours} h)")
    if getattr(args, "announce", None):
        args.announce(run_id)
    log_file = (run_dir / "container.log").open("w")
    container = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    sampler = MemorySampler(name, run_dir / SAMPLES_NAME)
    sampler.start()
    deadline = time.monotonic() + max_seconds + GRACE_SECONDS
    peak_sampled = 0
    stopped_early = False
    while True:
        try:
            container.wait(timeout=60)
            break
        except subprocess.TimeoutExpired:
            stats = docker("stats", "--no-stream", "--format", "{{.MemUsage}}", name, check=False)
            match = re.match(r"([\d.]+)\s*([KMG]i?B)", stats.stdout.strip())
            if match:
                factor = {"KB": 1e3, "KIB": 1024, "MB": 1e6, "MIB": 1024**2, "GB": 1e9, "GIB": 1024**3}[match.group(2).upper()]
                peak_sampled = max(peak_sampled, int(float(match.group(1)) * factor))
            if time.monotonic() > deadline:
                stopped_early = True
                docker("stop", "-t", "60", name, check=False)
                container.wait()
                break
    log_file.close()
    memory_samples = sampler.finish()
    exit_code = int(docker("inspect", "--format", "{{.State.ExitCode}}", name, check=False).stdout.strip() or -1)
    oom = docker("inspect", "--format", "{{.State.OOMKilled}}", name, check=False).stdout.strip() == "true"
    docker("rm", "-f", name, check=False)
    # Files made by the container's root belong to root on the host; hand them back.
    subprocess.run(["docker", "run", "--rm", "--entrypoint", "chown", "-v", f"{run_dir}:/w", args.image,
                    "-R", f"{os.getuid()}:{os.getgid()}", "/w"], capture_output=True)

    after = verify_data(data_mount, sums)
    audit = run_audit(args, run_dir, data_mount)
    manifest = read_json(run_dir / "records" / "run_manifest.json")
    error = read_json(run_dir / "records" / "error.json")
    failure, seen = decide_failure(manifest, error, exit_code, stopped_early, oom)
    usage = read_json(run_dir / "records" / "usage.json") or {}
    # The container's clock is UTC. Take start and end from the host's local time, like every other stamp.
    for key, source in (("start", "started_at"), ("end", "finished_at")):
        if manifest and manifest.get(source):
            usage[key] = datetime.fromisoformat(manifest[source]).astimezone().strftime("%Y-%m-%d %H:%M")
    usage.update(memory_samples)
    usage.update({
        "stopped_early": stopped_early, "data_unchanged_after_run": after["ok"],
        "data_check_detail": verified_text(after, mounted_read_only(data_mount)),
        "docker_stats_peak_sampled_gb_includes_cache": round(peak_sampled / 1024**3, 2) if peak_sampled else None,
        "oom_killed_by_docker": oom,
    })
    record.update({
        "status": "stopped_early" if stopped_early else "finished", "finished": local_stamp(),
        "stopped_early": stopped_early, "container_exit_code": exit_code,
        "limits_applied": usage.get("limits_applied"), "usage": usage,
        "data_check_after": after,
        "data_unchanged_after_run": after["ok"], "data_check_detail": usage["data_check_detail"],
        "audit": audit, "pipeline_status": (manifest or {}).get("run_status"),
        "failure_class": failure, "held_for_review": failure == "harness_suspect",
        "infrastructure_failure": failure if failure in ("infrastructure", "rate_limit") else None,
        "infrastructure_error_seen": seen,
        "counts_as_study_run": args.kind == "study" and failure is None,
    })
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=2))
    copy_results_safely(args, run_id, run_dir, record)
    print(f"{run_id}: pipeline {record['pipeline_status']}, audit {audit.get('summary')}, "
          f"{'counts as a study run' if record['counts_as_study_run'] else 'does not count: ' + str(failure or 'test run')}")
    return record


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def run_audit(args, run_dir: Path, data_mount: Path) -> dict:
    """scripts/audit_run.py inside the image, without network. Writes records/audit.json."""
    command = ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python",
               "--mount", f"type=bind,source={run_dir / 'out'},target=/audit/out,readonly",
               "--mount", f"type=bind,source={data_mount},target=/audit/data,readonly",
               "--mount", f"type=bind,source={run_dir / 'records'},target=/audit/records",
               args.image, "scripts/audit_run.py", "--run-dir", "/audit/out", "--data-dir", "/audit/data",
               "--json", "/audit/records/audit.json"]
    result = subprocess.run(command, capture_output=True, text=True)
    (run_dir / "records" / "audit.txt").write_text(result.stdout + result.stderr)
    last = [line for line in result.stdout.strip().splitlines() if line.strip()]
    return {"exit_code": result.returncode, "summary": last[-1] if last else "no output"}


def copy_results(args, run_id: str, run_dir: Path) -> None:
    """Study runs are copied after the run: out/ next to the AISS outputs, the records to the run-records folder.

    Test runs are copied only with --copy. Nothing is written to those folders while a run is going,
    and neither location is visible to agent runs.
    """
    if args.no_copy or (args.kind != "study" and not args.copy):
        return
    out_target = Path(args.out_copy_root) / run_id
    rec_target = Path(args.records_copy_root) / run_id
    # Agent code leaves symlinks (for example to the data folder inside the container). They point at
    # paths that do not exist here, and following them could copy data; so they are never copied.
    skip_symlinks = lambda folder, names: [n for n in names if (Path(folder) / n).is_symlink()]  # noqa: E731
    shutil.copytree(run_dir / "out", out_target / "out", dirs_exist_ok=True, ignore=skip_symlinks)
    shutil.copytree(run_dir / "records", rec_target / "records", dirs_exist_ok=True, ignore=skip_symlinks)
    for name in ("run_record.json", "container.log", SAMPLES_NAME):
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, rec_target / name)
    print(f"{run_id}: copied to {out_target} and {rec_target}")


def copy_results_safely(args, run_id: str, run_dir: Path, record: dict) -> None:
    """copy_results, but a failed copy is written into the run record instead of ending the queue.

    The run folder stays on disk, so a failed copy can be redone by hand.
    """
    try:
        copy_results(args, run_id, run_dir)
    except (OSError, shutil.Error) as error:
        record["copy_error"] = f"{type(error).__name__}: {str(error)[:500]}"
        (run_dir / "run_record.json").write_text(json.dumps(record, indent=2))
        print(f"{run_id}: COPY FAILED, the run folder is intact and needs copying by hand: {record['copy_error']}")


# ---------------------------------------------------------------- queue

NOT_STARTED_HEADER = "time,reason,started_by,run_id\n"


def log_not_started(args, reason: str, started_by: str, run_id: str = "") -> None:
    """One line per attempt that never became a run (or a folder found aborted), next to the copied records.

    Such attempts write no run record, so this file is what lets the paper's flow table count them.
    """
    path = Path(args.records_copy_root) / "not_started.csv"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if new:
                handle.write(NOT_STARTED_HEADER)
            writer.writerow([local_stamp(), " | ".join(reason.split("\n")), started_by, run_id])
    except OSError:
        pass


class Host:
    """What the queue asks the machine before each run. Replaced by a fake in the tests."""

    def active_run_names(self) -> list[str]:
        out = docker("ps", "--filter", "name=^ssa-ssa-", "--format", "{{.Names}}", check=False).stdout
        return [n for n in out.split() if n]

    def stale_container_names(self) -> list[str]:
        out = docker("ps", "-a", "--filter", "name=^ssa-ssa-", "--filter", "status=exited", "--format", "{{.Names}}", check=False).stdout
        return [n for n in out.split() if n]

    def available_memory_gb(self) -> float:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
        return float("inf")

    def free_disk_gb(self, path: Path) -> float:
        path = Path(path)
        while not path.exists() and path != path.parent:
            path = path.parent
        return shutil.disk_usage(path).free / 1024**3


def not_ready_reason(args, host) -> str | None:
    """Why a new run should not start right now. Waiting costs nothing; starting into these costs a run."""
    active = host.active_run_names()
    if active:
        return f"another SSA run is active on the same CPUs ({', '.join(active)})"
    if host.available_memory_gb() < args.min_free_gb:
        return f"only {host.available_memory_gb():.1f} GB of memory available (needs {args.min_free_gb:g})"
    for root in (args.runs_dir, args.out_copy_root, args.records_copy_root):
        if host.free_disk_gb(Path(root)) < args.min_free_disk_gb:
            return f"less than {args.min_free_disk_gb:g} GB free on the disk holding {root}"
    return None


def mark_orphans(runs_dir: Path, host, min_age_seconds: float = 600, on_marked=None) -> list[str]:
    """Run folders with no run_record.json whose container is gone were cut off (the script or the
    machine died). Write an "aborted" record so that no run silently disappears from the books."""
    active = {n[len("ssa-"):] for n in host.active_run_names()}
    marked = []
    for folder in sorted(Path(runs_dir).glob("ssa-*")):
        if not folder.is_dir() or (folder / "run_record.json").exists() or folder.name in active:
            continue
        if time.time() - folder.stat().st_mtime < min_age_seconds:
            continue
        (folder / "run_record.json").write_text(json.dumps({
            "run_id": folder.name, "system": "SSA", "status": "aborted", "counts_as_study_run": False,
            "failure_class": "aborted", "finished": local_stamp(),
            "note": "Found without a run record and without a running container: the script or the machine "
                    "stopped during the run. Written by the queue at its next start; the run was not completed.",
        }, indent=2))
        marked.append(folder.name)
        if on_marked:
            on_marked(folder.name)
    for name in host.stale_container_names():
        docker("rm", "-f", name, check=False)
    return marked


def write_status(path: Path, **fields) -> None:
    fields["updated"] = local_stamp()
    try:
        path.write_text(json.dumps(fields, indent=2, default=str))
    except OSError:
        pass


FAILURE_WAITS = [5 * 60, 15 * 60, 30 * 60]
SETUP_WAITS = [5 * 60, 15 * 60, 45 * 60, 60 * 60]


def run_queue(args, meta: dict, run=run_once, sleep=time.sleep, host=None) -> list[dict]:
    """Study runs one after another until args.n runs have counted (0 means until stopped).

    A run is never continued. A run that does not count (infrastructure failure) is replaced by a
    new one after a wait of 5, 15 and 30 minutes (a rate limit waits at least 15). The queue stops, and
    says why in queue_status.json, when: the STOP file exists (checked between runs and every minute while
    waiting), args.max_tokens is spent, the setup stays unready for args.setup_patience_hours, three
    runs in a row do not count, or two in a row are held as harness_suspect. Waiting for a free CPU
    range, memory or disk is not a failure.
    """
    host = host or Host()
    runs_dir = Path(args.runs_dir)
    status_path = runs_dir / "queue_status.json"
    stop_file = Path(args.stop_file) if args.stop_file else runs_dir / "STOP"
    records: list[dict] = []
    counted = tokens = consecutive = suspects = 0
    waited_for_setup = 0.0
    setup_attempts = 0
    mark_orphans(runs_dir, host, on_marked=lambda name: log_not_started(
        args, "aborted: run folder had no record and no running container; cause unknown", "queue", name))

    def state(message: str, **extra) -> None:
        write_status(status_path, state=message, counted=counted, attempts=len(records), tokens_total=tokens,
                     consecutive_not_counted=consecutive, held_for_review=suspects, **extra)
        print(message)

    def wait(seconds: float, why: str) -> bool:
        """Sleep in one-minute slices. False if the STOP file appeared."""
        state(f"Waiting {seconds / 60:g} min: {why}", waiting_until=local_stamp(datetime.now() + timedelta(seconds=seconds)))
        left = seconds
        while left > 0:
            if stop_file.exists():
                return False
            sleep(min(60, left))
            left -= 60
        return True

    while args.n == 0 or counted < args.n:
        if stop_file.exists():
            state(f"Stopped: {stop_file} exists.")
            break
        if args.max_tokens and tokens >= args.max_tokens:
            state(f"Stopped: token budget of {args.max_tokens:,} spent ({tokens:,}).")
            break
        reason = not_ready_reason(args, host)
        if reason:
            waited_for_setup += 300
            if waited_for_setup > args.setup_patience_hours * 3600:
                state(f"Stopped: not ready for {args.setup_patience_hours:g} h ({reason}).")
                break
            wait(300, reason)
            continue
        args.announce = lambda run_id: state(f"Running {run_id}", running=run_id)
        try:
            record = run(args, meta, attempt=consecutive + 1)
        except SetupError as error:
            log_not_started(args, str(error), "queue")
            wait_s = SETUP_WAITS[min(setup_attempts, len(SETUP_WAITS) - 1)]
            setup_attempts += 1
            waited_for_setup += wait_s
            if waited_for_setup > args.setup_patience_hours * 3600:
                state(f"Stopped: setup not ready for {args.setup_patience_hours:g} h: {error}", needs_attention=True)
                break
            state(f"Setup not ready ({error}). Nothing was started.", needs_attention=True)
            wait(wait_s, "setup not ready (VPN, login or image); a login needs Duke")
            continue
        except Exception as error:  # a bug must stop the queue visibly, not leave a stale "Running" status
            traceback.print_exc()
            state(f"Stopped: unexpected error in the queue ({type(error).__name__}: {str(error)[:200]}). "
                  "See the queue log.", needs_attention=True)
            break
        setup_attempts, waited_for_setup = 0, 0.0
        records.append(record)
        tokens += ((record.get("usage") or {}).get("tokens_total") or 0)
        failure = record.get("failure_class")
        if record["counts_as_study_run"]:
            counted += 1
            consecutive = suspects = 0
            state(f"{record['run_id']} counted ({counted} so far).", last_run=record["run_id"])
            continue
        consecutive += 1
        suspects = suspects + 1 if failure == "harness_suspect" else 0
        if suspects >= 2:
            state(f"Stopped: two runs in a row were held for review (last {record['run_id']}).", needs_attention=True, last_run=record["run_id"])
            break
        if consecutive >= args.max_consecutive_failures:
            state(f"Stopped: {consecutive} runs in a row did not count (last {record['run_id']}: {failure}).", needs_attention=True, last_run=record["run_id"])
            break
        pause = FAILURE_WAITS[min(consecutive - 1, len(FAILURE_WAITS) - 1)]
        if failure == "rate_limit":
            pause = max(pause, 15 * 60)
        wait(pause, f"{record['run_id']} did not count ({failure}); starting a replacement run afterwards")
    else:
        state(f"Done: {counted} runs counted.")
    return records


# ---------------------------------------------------------------- command line

def login(args) -> None:
    AZURE_DIR.mkdir(mode=0o700, exist_ok=True)
    tty = ["-it"] if sys.stdin.isatty() else []  # the device-code login needs no keyboard input
    subprocess.run(["docker", "run", *tty, "--rm", "--entrypoint", "az", "--mount",
                    f"type=bind,source={AZURE_DIR},target=/root/.azure", args.image, "login", "--use-device-code"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["login", "check", "run", "queue"])
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--prompt", help="the rendered study prompt file")
    parser.add_argument("--prompt-sha256", help="refuse to run if the prompt file has a different SHA-256")
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    parser.add_argument("--kind", choices=["study", "test"], default="test",
                        help="study: refused unless the study settings are met; test (default): never counts")
    parser.add_argument("--out-copy-root", default=str(OUT_COPY_ROOT), help="study runs: out/ is copied to <this>/<run_id>/out")
    parser.add_argument("--records-copy-root", default=str(RECORDS_COPY_ROOT), help="study runs: records are copied to <this>/<run_id>/")
    parser.add_argument("--copy", action="store_true", help="also copy a test run")
    parser.add_argument("--no-copy", action="store_true", help="copy nothing")
    parser.add_argument("--cpuset", default=CPUSET)
    parser.add_argument("--memory", default=MEMORY)
    parser.add_argument("--max-hours", type=float, default=4.0)
    parser.add_argument("--n", type=int, default=1, help="queue: number of counted study runs (0: until stopped)")
    parser.add_argument("--stop-file", help="queue: stop after the current run when this file exists (default <runs-dir>/STOP)")
    parser.add_argument("--max-tokens", type=int, default=0, help="queue: stop when all attempts together have used this many tokens (0: no limit)")
    parser.add_argument("--max-consecutive-failures", type=int, default=3, help="queue: stop after this many runs in a row that do not count")
    parser.add_argument("--setup-patience-hours", type=float, default=8.0, help="queue: stop if the setup or machine stays unready this long")
    parser.add_argument("--min-free-gb", type=float, default=8.0, help="queue: memory that must be available to start a run")
    parser.add_argument("--min-free-disk-gb", type=float, default=10.0, help="queue: free disk needed where runs and copies go")
    parser.add_argument("--fixture-literature", action="store_true", help="smoke tests only: skip the live literature search")
    args = parser.parse_args(argv)

    if args.command == "login":
        login(args)
        return 0
    if not args.prompt:
        parser.error("--prompt is required")
    try:
        meta = image_meta(args.dataset)
        if args.command == "check":
            check_setup(args, meta)
        elif args.command == "run":
            record = run_once(args, meta)
            return 0 if record["counts_as_study_run"] else 2
        else:
            args.kind = "study"
            records = run_queue(args, meta)
            return 0 if args.n and sum(1 for r in records if r["counts_as_study_run"]) >= args.n else 2
    except SetupError as error:
        print(f"Not started: {error}", file=sys.stderr)
        log_not_started(args, str(error), "queue" if args.command == "queue" else "manual")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
