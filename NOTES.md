# Working Notes

Running notes on setup, branch history, and findings. Written while getting the
pipeline running again in August 2026. Keep appending — this is the raw material
for the methods and limitations sections.

---

## Machine setup

### Environment variables (Windows, per machine)

| Variable | Value | Why |
|---|---|---|
| `AOAI_PRIVATE_IP` | the Azure private endpoint address | Read by `devcontainer.json` to add a hosts entry. **Container creation fails outright if this is unset** — Docker rejects `--add-host` with an empty IP. |

Set with `setx AOAI_PRIVATE_IP "<ip>"`, then reopen PowerShell *and* VS Code so
the value is inherited. Retrieve an existing value with:

```powershell
[Environment]::GetEnvironmentVariable("AOAI_PRIVATE_IP","User")
```

CPU limiting needs no variable. `.devcontainer/set-thread-limits.sh` runs at
container start, reads `nproc`, and caps the numeric thread pools at 85% —
17 threads on the 20-core desktop, 3 on the 4-core laptop.

### Prerequisites

Docker Desktop, VS Code + Dev Containers, Git, **Git LFS**, Google Cloud CLI.

Git LFS is not optional on this branch: the EHRSHOT data (~119 MB, 1,226 patient
files) is tracked through LFS. Cloning without it yields pointer files and a
confusing stage 3 failure.

The Azure CLI is *not* needed on the host — it's installed in the container image.

### Order of operations that actually works

1. `gcloud init`, then `gcloud auth application-default login`, then
   `gcloud auth application-default set-quota-project gac-som-dbmi-bpsmar-app-59`
2. Set `AOAI_PRIVATE_IP`, reopen PowerShell and VS Code
3. Connect GlobalProtect to `amc-vpn-ucdenver`; fully quit any personal VPN
4. Reopen in Container
5. In the container: `az login --use-device-code`, then set git identity
6. `python -m scripts.connectivity.test_azure` and `test_vertex`

Do steps 1–3 *before* first opening the container.

---

## Traps already hit (don't rediscover these)

**ADC file created as a directory.** If the container is opened before
`application_default_credentials.json` exists, Docker creates a *directory* at
that path. `gcloud auth application-default login` then fails with "Permission
denied," and the container fails to start with "not a directory: Are you trying
to mount a directory onto a file?" Fix: remove the container, delete the
directory, re-run the login, reopen.

**Stale container after fixing the above.** Deleting the host directory isn't
enough — the old container has `/tmp/google-adc.json` as a directory inside it.
`docker rm -f <id>`, then Reopen.

**`/app` was a frozen copy, not the repo.** The Dockerfile does `COPY . .` into
`/app`, and `devcontainer.json` originally set `workspaceFolder: /app` with no
matching `workspaceMount`. Edits made in the container never reached Windows and
vanished on rebuild. Fixed by adding:

```json
"workspaceMount": "source=${localWorkspaceFolder},target=/app,type=bind,consistency=cached"
```

This is why the old README insisted on committing before exiting.

**Azure 403 "Public access is disabled."** Not DNS, not VPN routing, not Docker.
The resource only accepts connections through a private endpoint that campus DNS
does not advertise — `bpsmar-ai-openai-1.privatelink.openai.azure.com` resolves
to a public gateway (20.119.156.99) even when querying campus DNS directly. IT
supplied the private IP; `devcontainer.json` adds it via `--add-host`. Note that
an unauthenticated request returns 401 while an authenticated one returns 403,
which is the signature of a network policy applied after auth.

**Personal VPN interference.** NordVPN takes the default route and DNS, silently
breaking the campus connection. Disconnecting isn't enough — quit the app.

**`devcontainer.json` changes need a container recreate.** `containerEnv`,
`runArgs`, and `workspaceMount` apply at creation. Committing changes nothing on
its own. Switching branches can also silently swap the file and strip these
settings — that happened once and cost an hour.

**Stale `.git/index.lock`.** Produces "Unable to create index.lock: File exists"
with no git process running. `rm -f /app/.git/index.lock`.

**VS Code disconnects kill runs.** Start long runs detached:
`nohup python main.py > run.log 2>&1 &`, watch with `tail -f run.log`. Survives
a disconnect, not a container restart.

---

## Branch situation

Three lineages matter:

- **`upstream/main`** — has the uncertainty work (`N_SAMPLES`, `sample_i/`
  folders) added 2026-07-24. **Cannot complete a run.** Stage 2 writes models to
  `generated_code/<run_id>/sample_0/<model>/`, but stage 3 looks in
  `generated_code/<run_id>/` and finds nothing. A second bug follows immediately:
  `benchmarking_sentences` calls `.model_dump_json()` on a LangChain state dict.
  Nothing after 07-24 touches the code — the later commits are file uploads.

- **`upstream/fix/agent-pipeline-reliability`** (Rayan Tahir) — branched from the
  last good commit on 07-22, before the uncertainty work. No `N_SAMPLES`, flat
  layout, paths consistent. Carries reliability patches verified across a 40-run
  temperature sweep: `recursion_limit` 150, `execute_python` cap raised 480s →
  900s, fuzzy model-name matching, model-folder dedup, rate-limit backoff.
  **This is what the team's actual experiments ran on.**

- **`working`** (local) — based on the fix branch, plus the devcontainer access
  fixes and the run logging described below.

The uncertainty feature appears never to have completed a run end to end. Worth
confirming with jcAlpaca before assuming.

### Minimal fix if returning to `main`'s lineage

Two one-liners, both in `main.py`:

```python
sample_dir = run_dir                            # was os.path.join(run_dir, f"sample_{i}")
str(result["messages"][-1].content)             # was result.model_dump_json()
```

The thorough version threads a sample directory through
`run_benchmarking_agent()`, `collect_benchmark_results()`, and
`collect_benchmark_scripts()` so `N_SAMPLES > 1` works properly.

---

## Findings about the pipeline itself

These matter for the paper, not just for getting it running.

**The literature agent is told almost nothing about the data.** The entire
description is "clinical data, specifically data in the format of EHRSHOT" and
"longitudinal EHRSHOT-style clinical data." It never learns the outcome is
binary, the cohort size, the class balance, that models receive a flat numeric
table of ~960 engineered features, or that they must expose sklearn-style
`fit`/`predict_proba`.

Consequence, observed directly: runs select DeepSurv, CLMBR-T-base, DuETT, and
RNN-for-EHR models — sequence and survival architectures — which are then handed
a static feature matrix and a binary label. First successful run produced AUROCs
of 0.508, 0.551, and 0.537. Essentially chance.

The team's sweep script worked around this by overriding the literature prompt to
exclude deep learning "so runs stay fast," which is a symptom of the same problem.

**Neither the feature matrix nor the split is actually frozen across runs.**

- `BenchmarkTaskConfig.split_file` is dead configuration. No repository code
  reads or writes it. Only the agent-generated `run_benchmark.py` touches it, and
  only to *write*. Nothing ever loads a previously frozen split.
- The split is computed from the **feature table**, not from `labels.csv`:
  `train_test_split(ids, y, test_size=..., random_state=seed, stratify=y)` where
  `ids` comes from `features['patient_id']`. So the cohort entering the split is
  whichever patients survived parsing. Change the parser, change the cohort,
  change the entire partition — seed 42 notwithstanding.
- The feature cache key is
  `md5({dataset, data_root, outcome, patient_id_column, feature_version})`, where
  `feature_version` is a constant the agent invents in its own generated code.
  A later run writing different code gets a different key, misses the cache, and
  re-parses with a different parser.

So full delegation introduces variance at four levels: which models are selected,
how they're implemented, what the features are, and **which patients land in the
test set**. The fourth is the one a reader would assume is controlled — two runs'
AUROCs are not measured on the same patients.

Every run overwrites `experiments/splits/ehrshot_new_hyperlipidemia.json`, so
snapshot it after each run to compare:

```bash
cp experiments/splits/ehrshot_new_hyperlipidemia.json /tmp/split_runN.json
```

**Roughly half of runs fail.** Matches what the team reported. The signature is
`RuntimeError: Agent never wrote .../benchmark_results.json`, typically after the
benchmarking agent has written `run_benchmark.py` but before producing
`benchmark_context.json` — it exhausts its self-correction steps or its script
errors and it gives up. Observed on the fix branch *with* `recursion_limit` at
150, so that patch reduces but does not eliminate it.

**The data is heterogeneous free text.** One row per encounter with a single text
cell holding varying measurements and results, so feature extraction is a parsing
problem solved fresh by each run's generated code. First run: 1,226 patients,
960 features, 96% parsing coverage, 20 missing event files.

---

## Run logging added

Two artifacts written per run, neither read by the pipeline, both surviving a
later-stage failure:

- `generated_code/<run_id>/literature.json` — the search agent's candidates in
  **ranked order**, with names, repo links, summaries, rationales. Rank is
  otherwise unrecoverable, since model folders are named by slug and read back
  alphabetically.
- `generated_code/<run_id>/code_generation.json` — per candidate: rank, name,
  family, and whether usable code was generated. Plus any extra models. Written
  *before* the count check that aborts runs, so failed runs still yield a
  success-rate record.

These exist because ~50% of runs die in benchmarking, and without them every
failed run loses its literature data entirely.

---

## Planned experiments

1. **AUROC vs. number of candidate models** (1 / 3 / 5). Hypothesis: more
   candidates means more poorly aligned methods, dragging the mean down.
   Confound to handle: the live literature search returns different models every
   run, so N and model identity vary together. Either fix the candidate pool
   (`load_base_literature()` or explicit `model_names`) to isolate the effect of
   count, or keep live search and report model identity as its own outcome.
   Report both mean and max AUROC — they answer different questions.
2. **Logistic regression behavior across agent temperature.** Anomaly already
   noticed by the group; Rayan's sweep tooling covers the temperature axis.
3. **Frozen versus agent-derived features.** The repo already contains both
   halves: `src/evaluation/deterministic.py` is a non-LLM evaluator with fixed
   features and splits, currently disabled and explicitly forbidden to the agent
   by the benchmarking prompt. Caveat: its feature builder is crude — per-column
   means, std, missingness, and *average string length* for text. Any frozen
   condition should reuse a real parser from a successful run
   (`generated_code/<run_id>/run_benchmark.py`) rather than that fallback.

---

## Open questions

**For the PI**
- EHRSHOT data is committed to this repository through git-LFS and is present in
  the upstream history. Is that consistent with the data use agreement, and is
  the upstream repo public or private? Someone removed data from `main` on
  07-29, which suggests the question was already raised.
- Should the literature prompt be given the data structure? It would very likely
  raise the success rate and AUROCs, but it changes the research question from
  "can an agent choose well unaided" to "can an agent benchmark well given an
  appropriate candidate pool."

**For jcAlpaca**
- Did a full run with `N_SAMPLES > 1` ever complete? Static reading says stage 3
  cannot find the models under `sample_i/`.

**For Rayan Tahir**
- Which reliability patches ended up mattering most across the 40-run sweep?
- Was the ~50% failure rate stable across temperatures?

**For the group**
- Which commit was each set of reported results produced on? Results from before
  07-24 and after are not from the same pipeline.

---

## Known issues still open

- `experiments/manifest.json` doesn't exist; `experiments/` is gitignored, so
  `scripts.experiment_runner` has no manifest to run.
- `experiment_runner.py` reads `run_manifest["leakage_passed"]`, which `main.py`
  no longer writes — will `KeyError`.
- Importing `src/uncertainty/uncertainty_quantification.py` runs three
  module-level demo prints, loading a sentence-transformer model on every start.
- Paths are container-absolute (`/app/...`) in many places; only some modules
  rewrite for host execution.
