"""User-editable configuration types for benchmarking experiments."""


from __future__ import annotations
from dataclasses import dataclass, field
import os

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
    "gemini-2.5-flash-lite",
)

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
   # How validation probabilities become binary diagnoses: "max_f1",
   # "minimum_sensitivity", or "cost".
   threshold_objective: str = "max_f1"
   minimum_sensitivity: float = 0.90
   false_positive_cost: float = 1.0
   false_negative_cost: float = 1.0


   def __post_init__(self) -> None:
       if self.dataset != "EHRSHOT":
           raise ValueError("The deterministic evaluator currently supports EHRSHOT only")
       if not 0 < self.test_fraction < 1 or not 0 <= self.validation_fraction < 1:
           raise ValueError("Invalid validation/test fractions")
       if self.test_fraction + self.validation_fraction >= 1:
           raise ValueError("Validation and test fractions must sum to less than 1")
       if self.threshold_objective not in {
           "max_f1",
           "minimum_sensitivity",
           "cost",
       }:
           raise ValueError(
               "threshold_objective must be max_f1, minimum_sensitivity, or cost"
           )
       if not 0 < self.minimum_sensitivity <= 1:
           raise ValueError("minimum_sensitivity must be in (0, 1]")
       if self.false_positive_cost < 0 or self.false_negative_cost < 0:
           raise ValueError("False-positive and false-negative costs cannot be negative")
       if self.false_positive_cost == self.false_negative_cost == 0:
           raise ValueError("At least one classification-error cost must be positive")
       if self.split_file is None:
           # Frozen per (dataset, outcome) so every run scores on the same
           # train/validation/test patients; mirrors the feature cache path.
           default_split_file = f"/app/experiments/splits/{self.dataset.lower()}_{self.outcome}.json"
           object.__setattr__(self, "split_file", default_split_file)
