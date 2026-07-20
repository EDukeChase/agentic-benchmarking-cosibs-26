# Linear Regression

Implementation uses `sklearn.linear_model.LinearRegression` as a linear-probability baseline for binary EHR outcomes.

## Decisions
- `predict_proba` is synthesized by clipping raw regression outputs to `[0, 1]` and returning `[1-p, p]`.
- `predict` thresholds the raw regression output at `0.5`.

## Assumptions
- Target labels are binary.
- Downstream evaluation can consume a 2-column probability array.

## Limitations
- Outputs are not true probabilities and may be poorly calibrated.
- Predictions outside `[0, 1]` are clipped for benchmarking convenience.
