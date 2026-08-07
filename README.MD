# agentic-benchmarking-ehr

A four-stage LLM agent pipeline that automates a clinical ML benchmarking study end to end. Given a prediction task on EHRSHOT data, the pipeline finds candidate models in the literature, implements them, benchmarks them, and writes up the results — then measures how much its own outputs vary between repeated runs.

---

## What the pipeline does

| Stage | Agent | Provider | Output |
|---|---|---|---|
| 1 | **Literature** | Gemini on Vertex AI, with Google Search grounding | *N* candidate models, each with a real paper and a real source-code repo |
| 2 | **Programming** | GPT on Azure OpenAI | A working `model.py` + `docs.md` per candidate |
| 3 | **Benchmarking** | GPT on Azure OpenAI, with a Python execution tool | Feature engineering, frozen splits, training, thresholding, metrics, per-patient predictions |
| 4 | **Reporting** | GPT on Azure OpenAI | A biostatistician-style narrative report (Markdown + JSON) |

Wrapped around all four:

- **Uncertainty quantification** — each stage runs `N_SAMPLES` times; outputs are embedded with a sentence transformer and the spread of those embeddings becomes that stage's uncertainty score.
- **Experiment runner** — repeats the whole pipeline across conditions and replicates, then runs ANOVA, Tukey HSD, and Lin's concordance across the results.

```
main.py
  │
  ├─ 1. literature    → LiteratureReviewResult (N candidates)
  ├─ 2. programming   → generated_code/<run_id>/sample_i/<model>/model.py, docs.md
  ├─ 3. benchmarking  → benchmark_results.json, predictions.json/.csv,
  │                     benchmark_context.json, test_<model>_benchmark.py
  └─ 4. reporting     → report.json, report.md, run_manifest.json
```

Everything lands in `generated_code/<run_id>/`, which is gitignored.

---

## Setup

Work through these in order. Steps 1–3 happen on **Windows**; steps 4–6 happen **inside the dev container**.

### Prerequisites (install these first)

| Software | Why | Where to get it |
|---|---|---|
| **Docker Desktop** | Runs the dev container | https://docs.docker.com/get-started/introduction/get-docker-desktop/ |
| **VS Code** + Dev Containers extension | Opens the project in the container | https://code.visualstudio.com/ |
| **Google Cloud CLI** | Creates the credentials the literature agent needs | https://cloud.google.com/sdk/docs/install |
| **GlobalProtect VPN** | Azure OpenAI is only reachable from the campus network | University IT |

Do **not** skip ahead and open the container before finishing steps 1–3. The container mounts a credentials file that has to exist first, and it needs an environment variable that has to be set first.

### 1. Google credentials

Run in **Windows PowerShell**:

```powershell
gcloud init
gcloud auth application-default login
```

During `gcloud init`, you can skip the default compute region/zone prompt — this project doesn't use Compute Engine.

On the consent screen, check **only** the first scope ("See, edit, configure, and delete your Google Cloud data..."). The Cloud SQL scope isn't needed.

Confirm you got a **file**, not a directory:

```powershell
Get-Item "$env:APPDATA\gcloud\application_default_credentials.json" | Select-Object Mode, Length
```

`Mode` should start with `-a`. If it starts with `d`, see [Troubleshooting](#troubleshooting).

### 2. Azure private endpoint address

The Azure OpenAI resource has public network access disabled — it only accepts connections through a private endpoint that campus DNS does not advertise. The container has to be told the address directly.

**Ask IT (or a teammate) for the private endpoint IP for `bpsmar-ai-openai-1.openai.azure.com`.** It is deliberately not stored in this repo.

Set it as a Windows environment variable:

```powershell
setx AOAI_PRIVATE_IP "10.x.x.x"
```

Then **close and reopen PowerShell and VS Code** so the new variable is picked up. Verify:

```powershell
$env:AOAI_PRIVATE_IP
```

`.devcontainer/devcontainer.json` reads this variable and adds a hosts entry inside the container. If the variable is empty, container creation will fail with an invalid IP error.

### 3. Connect to the VPN

Connect GlobalProtect to the **`amc-vpn-ucdenver`** profile. Nothing Azure-related works without it.

If you run a personal VPN (NordVPN, etc.), **quit it entirely** — not just disconnect. It will take over the default route and DNS and silently break the campus connection.

### 4. Open the dev container

In VS Code: `Ctrl+Shift+P` → **Dev Containers: Reopen in Container**.

First build takes a while (Python 3.14, ~100 pinned packages including torch, plus Azure CLI and git-lfs). Later starts are fast.

Confirm the setup landed:

```bash
grep bpsmar /etc/hosts     # should show your private endpoint IP
echo $GOOGLE_CLOUD_PROJECT # should print the project ID
```

### 5. Azure login

The Azure CLI is already installed in the image — nothing to install on Windows.

In the **container terminal** (not PowerShell):

```bash
az login --use-device-code
```

Use `--use-device-code`. Plain `az login` tries to open a browser the container doesn't have.

You'll need to repeat this whenever the token expires or the container is recreated.

### 6. Verify both clouds

Run from the repository root using module syntax. **Do not use the VS Code Run button** — it sets the working directory wrong and the `src.` imports break.

```bash
python -m scripts.connectivity.test_azure
python -m scripts.connectivity.test_vertex
```

`test_azure` should print a chat completion. `test_vertex` should end with `SUCCESS: Gemini used Google Search grounding.`

### 7. Data

**The EHRSHOT data is not in this repository.** It was removed from git history and is not redistributable. Ask the project owner for access.

Expected layout:

```
/app/data/EHR_SHOT/
  ├── labels.csv              # patient_id + outcome columns
  └── patient_data_all/
        ├── patient_1.csv     # one event file per patient
        └── ...
```

Stages 1–2 will run without it; stage 3 will fail.

---

## Running

```bash
git pull origin main
python main.py
```

Output goes to `generated_code/<run_id>/`, with `report.md` as the headline artifact. If a stage fails, `report.md` is still written and contains the stage name, error, and traceback — that's your main debugging surface.

### Before your first run, turn the cost down

The knobs are at the top of `main.py`:

```python
MODEL = "gpt-5.4"
TEMPERATURE = 1.0
NUMBER_OF_MODELS = 3     # candidates to find, implement, and benchmark
MAX_SEARCH_RESULTS = 1
N_SAMPLES = 5            # runs EVERY stage this many times, for uncertainty
```

`N_SAMPLES = 5` means five literature searches, five full code-generation runs, five benchmarking runs, and five report generations. **Set `N_SAMPLES = 1` and `NUMBER_OF_MODELS = 2` for a first smoke test.**

Stage timeouts default to 300s (900s for benchmarking), overridable via `PIPELINE_STAGE_TIMEOUT_SECONDS` and `BENCHMARK_STAGE_TIMEOUT_SECONDS`.

### Developing without burning API calls

`src/fixtures/base_literature.py` holds six hardcoded sklearn baselines. Uncomment the `load_base_literature` import in `main.py` to skip the literature agent entirely — stages 2–4 then get deterministic inputs, which makes debugging much faster and cheaper.

### Other scripts

```bash
python -m scripts.plot_predictions        # confusion-count bar chart per run
python -m scripts.experiment_runner experiments/manifest.json
```

Note that `experiments/` is gitignored, so no manifest ships with the repo — you'll need to write one. See [Known issues](#known-issues).

### Before you exit

The container is your dev environment; closing it ends the session.

```bash
git add -A && git commit -m "..." && git push
exit
```

Then bottom-left green badge → **Close Remote Connection**.

Git identity is per-container, so after a rebuild you'll need:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

---

## Project layout

| Path | Contents |
|---|---|
| `main.py` | Pipeline orchestrator. The `EXPERIMENT` block near the top is the main knob panel. Also handles stage timeouts, token accounting, and `run_manifest.json`. |
| `src/agents/authentication.py` | Azure bearer-token provider. Why `az login` is mandatory. |
| `src/agents/literature_agent.py` | Gemini + Google Search. Two calls: one grounded search, one schema-normalization pass (Vertex won't do both at once). |
| `src/agents/programming_agent.py` | `deepagents` over Azure GPT with a virtual filesystem. `_slugify()` defines the folder naming every later stage depends on. |
| `src/agents/benchmarking_agent.py` | Real filesystem plus an `execute_python` tool, followed by heavy post-hoc validation of the artifacts the agent was told to produce. |
| `src/agents/reporting_agent.py` | Joins all artifacts into report entries, then a structured-output LLM writes the narrative. Optional self-consistency judging. |
| `src/settings/config.py` | `LLMConfig`, `SelfConsistencyConfig`, `ExperimentConfig`, and `BenchmarkTaskConfig` (the frozen task: outcome, seed, split fractions, threshold objective). |
| `src/settings/prompts.py` | All four system prompts. The benchmarking prompt is effectively the statistical analysis plan — read it first when results look wrong. |
| `src/core/schemas.py` | Every Pydantic contract in one file. Fastest way to understand the data flow. |
| `src/evaluation/benchmark_tools.py` | Reads and validates the agent's result artifacts. |
| `src/evaluation/deterministic.py` | Repo-owned non-LLM evaluator. Currently disabled in `main.py`; kept as a reproducible control. |
| `src/evaluation/statistical_analysis.py` | Descriptive stats, ANOVA, Tukey HSD, Lin's concordance across experiment results. |
| `src/uncertainty/uncertainty_quantification.py` | Embedding-based spread across repeated stage outputs. |
| `src/reporting/markdown_report.py` | Report rendering and failure reports. |
| `src/fixtures/base_literature.py` | Hardcoded sklearn baselines for offline development. |
| `src/utils/telemetry.py` | Best-effort token accounting across LangChain response shapes. |
| `scripts/` | Experiment runner, prediction plots, connectivity tests. |
| `host.json`, `.vscode/` | Leftover Azure Functions scaffolding. Not used by the pipeline. |

### Changing the task

`BenchmarkTaskConfig` in `src/settings/config.py` defines the frozen task. Available outcomes: `new_acutemi`, `new_celiac`, `new_hyperlipidemia`, `new_hypertension`, `new_lupus`, `new_pancan`.

All candidate models in a run are evaluated against the same outcome, feature matrix, seed, and persisted patient-level split. Generated agents do not choose their own dataset, target, or split.

---

## Troubleshooting

**`gcloud auth application-default login` fails with "Permission denied"**
Docker created a *directory* where the credentials file should be, because the container was opened before the file existed. Close the container, remove it (`docker rm -f <id>`), delete the directory (`Remove-Item "$env:APPDATA\gcloud\application_default_credentials.json" -Recurse -Force`), re-run the login, then reopen.

**Container won't start: "not a directory: Are you trying to mount a directory onto a file?"**
A stale container from before the credentials file existed. `docker rm -f <id>`, then Reopen in Container.

**`403 - Public access is disabled. Please configure private endpoint.`**
The hosts entry isn't in place. Check `grep bpsmar /etc/hosts`. If it's missing, `AOAI_PRIVATE_IP` wasn't set when the container was created — set it, `docker rm -f <id>`, and reopen. As a one-off you can add it manually, but Docker regenerates `/etc/hosts` on restart:

```bash
echo "<private-ip>  bpsmar-ai-openai-1.openai.azure.com" >> /etc/hosts
```

**Azure calls hang or time out**
VPN. Confirm GlobalProtect is on the `amc-vpn-ucdenver` profile and any personal VPN is fully quit.

**`KeyError: 'GOOGLE_CLOUD_PROJECT'`**
The variable isn't set. It's in `containerEnv` now, so recreate the container — or `export GOOGLE_CLOUD_PROJECT=gac-som-dbmi-bpsmar-app-59` for the session.

**Changes to `devcontainer.json` don't take effect**
`containerEnv` and `runArgs` apply at container *creation*. Committing the file changes nothing on its own. `docker rm -f <id>`, then Reopen.

**Import errors like `ModuleNotFoundError: No module named 'src'`**
You're not in the repo root, or you used the VS Code Run button. Run from `/app` with `python -m scripts.x`.

**Diverged from `origin/main`**
This repo has a history of force-pushes. Always `git pull origin main` before working. To recover: copy your work outside the repo, then `git fetch origin`, `git reset --hard origin/main`, `git clean -fd`.

---

## Known issues

- **`experiments/manifest.json` doesn't exist.** `experiments/` is gitignored, so the experiment runner has no manifest to run. It needs `experiment_id`, `task`, `baseline`, and `conditions` (each with `condition_id`, `overrides`, `repetitions`).
- **`experiment_runner.py` will `KeyError` on `leakage_passed`.** It reads that key from `run_manifest.json`, but `main.py` no longer writes it — the leakage audit lived in the deterministic evaluator, which is currently disabled.
- **Importing `src/uncertainty/uncertainty_quantification.py` runs a demo.** Three module-level `print(calculate_uncertainty(...))` calls fire on import, loading a sentence-transformer model every time `main.py` starts.
- **Benchmarking samples overwrite each other.** With `N_SAMPLES > 1`, all benchmarking runs share one `run_id`, so collected artifacts come from the *last* sample while the response fed into the report comes from the *first*. See the TODO in `main.py`.
- **Paths are container-absolute.** Lots of hardcoded `/app/...`. Some modules rewrite it for host execution, not all. Stay in the container.
