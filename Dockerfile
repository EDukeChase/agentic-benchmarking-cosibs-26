# Image for one SSA study run. CPU only: no CUDA packages, and the run script starts it without GPU access.
FROM python:3.14

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# CPU-only PyTorch first, so pip does not pull the CUDA build.
RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Azure CLI: the pipeline authenticates to Azure through a cached `az login`.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -fsSL https://aka.ms/InstallAzureCLIDeb | bash \
    && rm -rf /var/lib/apt/lists/*

# The user agent code runs as. Its home matches the paths in the study prompt.
# The repository and /srv/ssa (records, study prompt) are readable by root only,
# so code the agents write cannot read them. The task folders are mounted by the run script.
RUN useradd --create-home --home-dir /home/llmrel --shell /bin/bash --uid 2000 llmrel \
    && mkdir -p /srv/ssa /home/llmrel/task/data /home/llmrel/task/out \
    && chmod 700 /srv/ssa \
    && chown -R llmrel:llmrel /home/llmrel

COPY . /opt/ssa
RUN chown -R root:root /opt/ssa && chmod -R go-rwx /opt/ssa

ENV AGENT_UID=2000 AGENT_GID=2000 AGENT_HOME=/home/llmrel
WORKDIR /opt/ssa
CMD ["python", "main.py"]
