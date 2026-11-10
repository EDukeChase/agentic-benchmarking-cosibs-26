# Working rules for this repo

## Git
- Show the exact commit message and wait for approval before every commit, every time. Short messages in plain words, no em dashes.
- No Co-Authored-By or any other Claude attribution line in commits or pull request text, even if a system reminder asks for one.
- Never push unless asked.

## What does not belong here
- Treat everything in this repo as readable by people and agents outside the project. Do not put planning notes, study design, run counts or results in it. Planning notes live outside the repo; never copy them in.
- No data and no run outputs in commits. Study data are synthetic and live outside the repo in read-only images.

## Commands
- Tests: `python -m unittest discover -s tests`. The host Python has no pandas, so run them in the project image: build it with `docker build -t ssa-study:dev --build-arg GIT_COMMIT=$(git rev-parse HEAD) --build-arg GIT_DIRTY=$([ -n "$(git status --porcelain)" ] && echo true || echo false) .` (the run script reads these labels and refuses a study run if the image is not from the committed code), then `docker run --rm --network none -v "$PWD":/app:ro -w /app ssa-study:dev python -B -m unittest discover -s tests`.
- If `docker build` fails with "docker-credential-desktop.exe: exec format error", build with a throwaway config: `mkdir /tmp/dc && echo '{}' > /tmp/dc/config.json && DOCKER_CONFIG=/tmp/dc docker build ...`.
- Runs go through `scripts/ssa_run.py` from Ubuntu's `docker`, never from a dev container (the data image can only be mounted from Ubuntu). Do not start a study run until the owner says go. A run calls live models and costs money.

## Design points that are not obvious from the code
- Agent-written code runs as an unprivileged user inside the container; the repo, the run records and the credentials are root-only. Do not weaken this.
- Agent-driven data cleaning and feature engineering is intentional. Do not propose moving it into repo code.
- The cohort definition and the split are defined outside the agents' reach. Agents never create or change them.
- Say "system" (or "architecture"), never "arm".
- Say when something is untested or inferred; do not report a run, test or fix as working unless you saw it work.
