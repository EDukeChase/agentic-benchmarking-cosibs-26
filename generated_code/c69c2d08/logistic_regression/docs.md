# Logistic Regression

Implementation notes:
- Wraps `sklearn.linear_model.LogisticRegression` with a small benchmarking API.
- Supports the main regularization and solver settings exposed by the wrapper.
- Provides `predict_proba`, `decision_function`, and a small binary evaluation helper.

Assumptions made:
- The literature review did not specify a canonical solver, penalty, or class
  weighting strategy. The implementation exposes these as configurable fields and
  defaults to scikit-learn's standard `lbfgs` + `l2` setup.
- The review described a binary baseline, so `evaluate_binary` reports accuracy
  and ROC AUC when the labels are binary.

Limitations:
- No preprocessing, calibration, or threshold tuning is included.
- Multiclass evaluation helpers are not provided, though the underlying model can
  handle multiclass if scikit-learn supports the chosen solver/penalty.
- Missing data must be imputed upstream.
