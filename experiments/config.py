"""Configuration for the celiac disease logistic regression baseline."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LogisticRegressionConfig:
    """Hyperparameters and runtime options for the baseline model.

    Assumptions
    ----------
    - Input features are provided as a 2D array-like object with shape
      ``(n_samples, n_features)``.
    - Targets are binary labels encoded as ``0`` and ``1``.
    - Standardization is applied within a scikit-learn pipeline to avoid data
      leakage.
    - The positive class is label ``1`` unless explicitly documented otherwise.
    """

    C: float = 1.0
    max_iter: int = 1000
    penalty: str = "l2"
    solver: str = "lbfgs"
    random_state: Optional[int] = 42
    class_weight: Optional[str] = None
    n_jobs: Optional[int] = None
    tol: float = 1e-4
    fit_intercept: bool = True
    verbose: int = 0
    standardize: bool = True
    positive_class: int = 1
    threshold: float = 0.5
    pipeline_name: str = field(default="logistic_regression_baseline", init=False)

