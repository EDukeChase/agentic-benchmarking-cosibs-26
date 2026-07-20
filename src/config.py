"""User-editable configuration types for benchmarking experiments."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMConfig:
    """Settings shared by one pipeline stage's OpenAI chat model."""

    model: str = "gpt-5.4-mini"
    temperature: float = 0.0
    timeout: int = 120
    max_retries: int = 2


@dataclass(frozen=True)
class SelfConsistencyConfig:
    """Generate several reports and use a judge model to combine them."""

    samples: int = 1
    model: str = "gpt-5.4-mini"
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise ValueError("Self-consistency samples must be at least 1")


@dataclass(frozen=True)
class ExperimentConfig:
    number_of_models: int = 5
    max_search_results: int = 1
    literature_llm: LLMConfig = field(default_factory=LLMConfig)
    programming_llm: LLMConfig = field(default_factory=LLMConfig)
    benchmarking_llm: LLMConfig = field(default_factory=LLMConfig)
    reporting_llm: LLMConfig = field(default_factory=LLMConfig)
    self_consistency: SelfConsistencyConfig = field(
        default_factory=SelfConsistencyConfig
    )


@dataclass(frozen=True)
class BenchmarkTaskConfig:
    """Frozen task definition shared by every model and experimental condition."""

    dataset: str = "EHRSHOT"
    data_root: str = "/app/data/EHR_SHOT"
    outcome: str = "new_celiac"
    patient_id_column: str = "patient_id"
    seed: int = 42
    test_fraction: float = 0.20
    validation_fraction: float = 0.20
    split_file: str = "/app/experiments/splits/ehrshot_celiac_seed42.json"

    def __post_init__(self) -> None:
        if self.dataset != "EHRSHOT":
            raise ValueError("The deterministic evaluator currently supports EHRSHOT only")
        if not 0 < self.test_fraction < 1 or not 0 <= self.validation_fraction < 1:
            raise ValueError("Invalid validation/test fractions")
        if self.test_fraction + self.validation_fraction >= 1:
            raise ValueError("Validation and test fractions must sum to less than 1")
