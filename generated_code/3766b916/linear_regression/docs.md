# Linear Regression

## Implementation notes
- Uses `sklearn.linear_model.LinearRegression` as an ordinary least squares baseline.
- Treats the regression output as a linear-probability score for binary EHR outcomes.
- Provides `predict_scores`, `predict`, and `evaluate` helpers in one importable module.

## Assumptions
- Features are preprocessed into numeric vectors.
- Labels are encoded as 0/1 for binary benchmarking.
- Raw predictions are clipped to `[0, 1]` before thresholding, because the literature review notes that the model is not probability constrained.

## Known limitations
- This is not a calibrated probabilistic classifier.
- Performance can be unstable when the true target is strongly nonlinear or imbalanced.
- AUROC is computed from clipped scores; if unclipped scores are preferred, that behavior should be changed explicitly.
