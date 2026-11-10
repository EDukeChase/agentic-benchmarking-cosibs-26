"""Run agent-written code as an unprivileged user.

Inside the study container the pipeline runs as root, so it can read the
credentials, the repository and its own run records. Code the agents write runs
as a separate user that can read none of those: the repository and the records
are root-only, and the process gets a minimal environment with no credentials.

If AGENT_UID is not set (development, unit tests) code runs as the current user.
This module does not make a sandbox by itself; the container image sets up the
users and the file permissions (see the Dockerfile).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

# Only these variables reach agent code. Credentials and endpoints are not among them.
PASSED_ENV = (
    "PATH", "LANG", "LC_ALL", "TZ",
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
)


@dataclass(frozen=True)
class AgentIdentity:
    uid: int
    gid: int
    home: str


def agent_identity() -> AgentIdentity | None:
    """The user agent code runs as, or None to run as the current user."""
    uid = os.getenv("AGENT_UID")
    if uid is None:
        return None
    return AgentIdentity(uid=int(uid), gid=int(os.getenv("AGENT_GID", uid)), home=os.getenv("AGENT_HOME", "/home/llmrel"))


def agent_env(identity: AgentIdentity | None) -> dict[str, str]:
    env = {key: os.environ[key] for key in PASSED_ENV if key in os.environ}
    env["HOME"] = identity.home if identity else os.environ.get("HOME", "/tmp")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_as_agent(args: list[str], cwd: str, timeout: float) -> subprocess.CompletedProcess:
    """subprocess.run as the agent user, with the minimal environment."""
    identity = agent_identity()
    kwargs = {"user": identity.uid, "group": identity.gid, "extra_groups": []} if identity else {}
    return subprocess.run(
        args, cwd=cwd, env=agent_env(identity), capture_output=True, text=True, timeout=timeout, **kwargs,
    )
