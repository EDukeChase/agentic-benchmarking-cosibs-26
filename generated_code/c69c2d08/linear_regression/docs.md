# Linear Regression

Implementation notes:
- Wraps `sklearn.linear_model.LinearRegression`.
- Intended as a simple baseline for continuous outcomes or linear-probability
  scoring for binary outcomes.
- Binary predictions are produced by clipping raw regression outputs to `[0, 1]`
  and thresholding at `0.5`.

Assumptions made:
- The literature review did not specify preprocessing, missing-value handling,
  or evaluation metrics, so this module only handles numeric array-like inputs
  and provides minimal evaluation helpers.
- For binary classification, the model is not a calibrated probabilistic model;
  clipped predictions are only a heuristic score.

Limitations:
- No feature preprocessing, scaling, or imputation.
- No support for sparse input-specific optimizations beyond what scikit-learn
  already provides.
- No automatic label validation beyond simple integer casting in binary eval.
