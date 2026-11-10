"""Configuration for the study pipeline: models, task folders and time limits."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

GOOGLE_CLOUD_PROJECT = os.getenv(
    "GOOGLE_CLOUD_PROJECT",
    "gac-som-dbmi-bpsmar-app-59",
)

GOOGLE_CLOUD_LOCATION = os.getenv(
    "GOOGLE_CLOUD_LOCATION",
    "global",
)

LITERATURE_MODEL = os.getenv(
    "LITERATURE_MODEL",
    "gemini-4-argon",
)

MODEL = "gpt-6.1-sol"
TEMPERATURE = 1.0
# Azure calls this reasoning_effort (low/medium/high/xhigh); Vertex calls it
# thinking_level. The study uses "high" for every stage.
REASONING_EFFORT = "high"
# Methods 3 to 5 of the study prompt are chosen by the literature stage. Methods
# 1 and 2 (the two logistic regressions) are defined by the study prompt itself.
NUMBER_OF_MODELS = 3
SEED = 5960

# Seconds. The whole pipeline has a hard limit (the study prompt tells the agents
# 4 hours). Each stage also gets its own cap, and benchmarking gets whatever is
# left after the reporting reserve is set aside.
TOTAL_DEADLINE_SECONDS = 4 * 60 * 60
LITERATURE_STAGE_SECONDS = 20 * 60
PROGRAMMING_STAGE_SECONDS = 60 * 60
REPORTING_STAGE_SECONDS = 15 * 60
# Longest single execute_python call the benchmarking agent may make.
EXECUTE_PYTHON_MAX_SECONDS = 30 * 60


@dataclass(frozen=True)
class LLMConfig:
    """Settings shared by one pipeline stage's chat model."""

    model: str = MODEL
    temperature: float = TEMPERATURE
    timeout: int = 600
    max_retries: int = 5
    reasoning_effort: str | None = REASONING_EFFORT


@dataclass(frozen=True)
class ExperimentConfig:
    number_of_models: int = NUMBER_OF_MODELS
    literature_llm: LLMConfig = field(
        default_factory=lambda: LLMConfig(model=LITERATURE_MODEL)
    )
    programming_llm: LLMConfig = field(default_factory=LLMConfig)
    benchmarking_llm: LLMConfig = field(default_factory=LLMConfig)
    reporting_llm: LLMConfig = field(default_factory=LLMConfig)


@dataclass(frozen=True)
class StudyTaskConfig:
    """Where the task lives. The study prompt is rendered for exactly these paths.

    ``data_dir`` and ``out_dir`` are the only places the agents may read and
    write, and ``task_root`` (their common parent) is the root of the agents'
    file tools. ``records_dir`` holds the pipeline's own records (manifest,
    literature result, split check) and is NOT under ``task_root``, so agents
    cannot see it.
    """

    dataset: str = "study_v1"
    data_dir: str = "/home/llmrel/task/data"
    out_dir: str = "/home/llmrel/task/out"
    records_dir: str = "/srv/ssa/records"
    study_prompt_file: str = "/srv/ssa/study_prompt.md"
    seed: int = SEED

    def __post_init__(self) -> None:
        if self.seed != SEED:
            raise ValueError(f"The study seed is fixed at {SEED}")

    @property
    def task_root(self) -> str:
        return os.path.commonpath([self.data_dir, self.out_dir])

    @property
    def methods_dir(self) -> str:
        """Where the programming stage writes the code for methods 3 to 5."""
        return os.path.join(self.out_dir, "code", "methods")

    @classmethod
    def from_env(cls) -> "StudyTaskConfig":
        defaults = cls()
        return cls(
            dataset=os.getenv("STUDY_DATASET", defaults.dataset),
            data_dir=os.getenv("STUDY_DATA_DIR", defaults.data_dir),
            out_dir=os.getenv("STUDY_OUT_DIR", defaults.out_dir),
            records_dir=os.getenv("STUDY_RECORDS_DIR", defaults.records_dir),
            study_prompt_file=os.getenv("STUDY_PROMPT_FILE", defaults.study_prompt_file),
        )
