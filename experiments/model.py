"""Model architecture definition for the celiac disease logistic regression baseline."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import LogisticRegressionConfig


def build_model(config: LogisticRegressionConfig) -> Pipeline:
    """Build a standardization + logistic regression pipeline.

    Notes
    -----
    - Standardization is performed inside the pipeline to prevent leakage.
    - The scikit-learn solver is configurable, but the literature review
      specifically calls for L2-regularized logistic regression.
    """

    steps = []
    if config.standardize:
        steps.append(("scaler", StandardScaler()))

    clf = LogisticRegression(
        C=config.C,
        max_iter=config.max_iter,
        penalty=config.penalty,
        solver=config.solver,
        random_state=config.random_state,
        class_weight=config.class_weight,
        n_jobs=config.n_jobs,
        tol=config.tol,
        fit_intercept=config.fit_intercept,
        verbose=config.verbose,
    )
    steps.append(("logreg", clf))
    return Pipeline(steps=steps)

