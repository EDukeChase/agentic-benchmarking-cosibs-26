To be able to use this repo for code, you need to have docker installed and the docker app running on you computer.
For docker installation, here is the link to download docker desktop: https://docs.docker.com/get-started/introduction/get-docker-desktop/

Now that you have docker installed, when you open up your IDE, you can have the option to open this project as a dev container. First open up the docker desktop app, then do that.

Alternative: open PowerShell in the project directory and build the image:

```powershell
docker build -t benchmark-image .
```

After running `gcloud auth application-default login` on Windows, start the
container with both the repository and your ADC file mounted:

```powershell
docker run --rm -it `
  --mount "type=bind,source=$PWD,target=/app" `
  --mount "type=bind,source=$env:APPDATA\gcloud\application_default_credentials.json,target=/tmp/google-adc.json,readonly" `
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google-adc.json `
  benchmark-image bash
```

Now you are in a fully virtual environment with all the dependencies you need for development, enjoy!

For actually working with the agents, the first thing you have to do is connect to the VPN (specifically the amc-vpn-ucdenver one)
Next, you have to type "az login" in the command line, and it will log you in automatically.
You can test Azure connectivity with `python -m scripts.connectivity.test_azure`.
Use `python -m scripts.connectivity.test_vertex` to test Vertex AI and Google
Search grounding. Run these commands from the repository root rather than using
the VS Code run-code button.

For coding in this environment, PLEASE ALWAYS RUN "git pull origin main" BEFORE DOING ANY CODING. IF YOU FORGET, YOU WILL HAVE TO MOVE YOUR CODE THAT YOU JUST MADE OUT OF THE REPOSITORY TO SOMEWHERE SAFE AND RUN "git fetch origin" then "git reset --hard origin/main" then "git clean -fd."

Before exiting, make sure to commit and push any changes first, as exiting the docker container closes your development environment.
You can exit by typing "exit" in the command line, and by clicking in the bottom left of your vscode and clicking the "Close Remote Connection" option.

## Project layout

- `src/agents/` — literature, programming, benchmarking, and reporting agents
- `src/settings/` — user-editable configuration and prompts
- `src/core/` — shared Pydantic schemas
- `src/evaluation/` — benchmark collection, deterministic evaluation, and statistics
- `src/reporting/` — Markdown and PDF report generation
- `src/fixtures/` — local fallback literature data
- `src/utils/` — shared utilities such as token accounting
- `scripts/` — experiment and prediction-artifact commands
- `scripts/connectivity/` — manual Azure and Vertex AI connection tests

Run repository scripts from the project root using module syntax, for example:

```bash
python -m scripts.plot_predictions
python -m scripts.export_predictions_csv
```

## Changing benchmark settings

The main experiment controls are grouped in the `EXPERIMENT` block near the top
of `main.py`. From there you can change:

- the number of candidate models and search results;
- the OpenAI model and temperature used by each pipeline stage; and
- the number of self-consistency report samples and the judge model.

Editable defaults and system prompts are in `src/settings/config.py` and
`src/settings/prompts.py`. `main.py` assigns each prompt
to a named variable, which can also be replaced with a prompt variant for an
individual experiment.

Self-consistency is disabled when `samples=1`, preserving the original behavior.
When `samples` is greater than one, the reporting model produces that many
independent narratives and the configured self-consistency model combines them
into the final report. For useful variation, set the reporting model temperature
above zero. Be aware that increasing `samples` also increases API usage.

## Reproducible experiments

`experiments/manifest.json` defines the frozen benchmark task, baseline settings,
experimental conditions, and number of independent full-pipeline repetitions.
Run it inside the project container with:

```bash
python -m scripts.experiment_runner experiments/manifest.json
```

Each independent pipeline run writes `run_manifest.json`, provider-reported token
usage when available, stage and total runtime, deterministic patient-level
predictions, a leakage audit, and benchmark results under its generated-code run
folder. Consolidated rows and SAP-style descriptive statistics, one-way ANOVA,
Tukey HSD comparisons, and pairwise Lin concordance coefficients are written to
`experiments/results/<experiment_id>/`.

All candidate models in a run are evaluated by `src/evaluation/deterministic.py`
against the same EHRSHOT outcome, engineered feature matrix, seed, and persisted
patient-level train/validation/test split. Generated agents no longer choose their
own dataset, target, or split. Full-pipeline self-consistency is controlled by each
condition's `repetitions`; `SelfConsistencyConfig.samples` remains only an optional
report-narrative sampling feature.
