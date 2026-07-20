# Linear Regression

Implementation: scikit-learn `LinearRegression` used as a linear-probability baseline.

Assumptions:
- Binary targets are expected.
- Raw regression outputs are clipped to `[0, 1]` for `predict_proba`.
- No imputation or scaling is applied inside this module.

Limitations:
- Outputs are not intrinsically calibrated probabilities.
- May produce poor performance on nonlinear EHR relationships.
