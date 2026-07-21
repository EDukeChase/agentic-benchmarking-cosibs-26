"""User-editable configuration types for benchmarking experiments."""


from __future__ import annotations


from dataclasses import dataclass, field

MODEL = "gpt-5.4"
TEMPERATURE = 1.0
NUMBER_OF_MODELS = 3
MAX_SEARCH_RESULTS = 1



@dataclass(frozen=True)
class LLMConfig:
   """Settings shared by one pipeline stage's OpenAI chat model."""


   model: str = MODEL
   temperature: float = TEMPERATURE
   timeout: int = 600
   max_retries: int = 2




@dataclass(frozen=True)
class SelfConsistencyConfig:
   """Generate several reports and use a judge model to combine them."""


   samples: int = 1
   model: str = MODEL
   temperature: float = TEMPERATURE


   def __post_init__(self) -> None:
       if self.samples < 1:
           raise ValueError("Self-consistency samples must be at least 1")




@dataclass(frozen=True)
class ExperimentConfig:
   number_of_models: int = NUMBER_OF_MODELS
   max_search_results: int = 1
   literature_llm: LLMConfig = field(default_factory=LLMConfig)
   programming_llm: LLMConfig = field(default_factory=LLMConfig)
   benchmarking_llm: LLMConfig = field(default_factory=LLMConfig)
   reporting_llm: LLMConfig = field(default_factory=LLMConfig)
   self_consistency: SelfConsistencyConfig = field(
       default_factory=SelfConsistencyConfig
   )


# "new_acutemi","new_celiac","new_hyperlipidemia","new_hypertension","new_lupus","new_pancan"
@dataclass(frozen=True)
class BenchmarkTaskConfig:
   """Frozen task definition shared by every model and experimental condition."""


   dataset: str = "EHRSHOT"
   data_root: str = "/app/data/EHR_SHOT"
   outcome: str = "new_hyperlipidemia"
   patient_id_column: str = "patient_id"
   seed: int = 42
   # percentage of the test and validation (how they are divided)
   test_fraction: float = 0.20
   validation_fraction: float = 0.20
   split_file: str | None = None


   def __post_init__(self) -> None:
       if self.dataset != "EHRSHOT":
           raise ValueError("The deterministic evaluator currently supports EHRSHOT only")
       if not 0 < self.test_fraction < 1 or not 0 <= self.validation_fraction < 1:
           raise ValueError("Invalid validation/test fractions")
       if self.test_fraction + self.validation_fraction >= 1:
           raise ValueError("Validation and test fractions must sum to less than 1")
       if self.split_file is None:
           # Frozen per (dataset, outcome) so every run scores on the same
           # train/validation/test patients; mirrors the feature cache path.
           default_split_file = f"/app/experiments/splits/{self.dataset.lower()}_{self.outcome}.json"
           object.__setattr__(self, "split_file", default_split_file)
