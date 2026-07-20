# Logistic Regression

Implementation uses `sklearn.linear_model.LogisticRegression`.

## Decisions
- Exposes both `predict` and `predict_proba` for benchmarking.
- Uses a high default `max_iter` to reduce convergence failures.

## Assumptions
- Binary labels are used in the main benchmark setting.

## Limitations
- Solver/regularization details are left configurable via `**kwargs`.
