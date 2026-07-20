# Linear Regression

## Implementation decisions
- Uses `sklearn.linear_model.LinearRegression` as the core estimator.
- Exposes regression outputs directly via `predict`.
- Adds `predict_proba` and `predict_label` helpers for binary-outcome benchmarking.
- Supports `sample_weight` during fitting and scoring through scikit-learn.

## Assumptions
- The review frames linear regression as a binary baseline even though the model is inherently a regression model.
- Probabilities are not native to linear regression, so `predict_proba` clips predictions to `[0, 1]` before forming a two-column output.
- Thresholding for binary labels uses `0.5`.

## Limitations
- Output is not calibrated.
- Multi-class classification is not supported by the helper methods.
- If labels are not numeric or not binary, classification convenience methods may be inappropriate.
