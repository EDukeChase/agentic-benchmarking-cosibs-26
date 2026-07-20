# Logistic Regression

## Implementation decisions
- Uses `sklearn.linear_model.LogisticRegression` as the core estimator.
- Exposes `predict`, `predict_proba`, and `decision_function` directly.
- Includes a simple binary evaluation helper returning accuracy, AUROC, and AUPRC.
- Supports `sample_weight` during fitting and scoring through scikit-learn.

## Assumptions
- The review describes a binary classification baseline.
- Default configuration uses `penalty='l2'` and `solver='lbfgs'`, which are broadly compatible and common defaults.
- If a caller needs sparse or L1/elastic-net behavior, they must set compatible solver/penalty options explicitly.

## Limitations
- Multiclass handling is delegated to scikit-learn and is not specialized here.
- No preprocessing, calibration, or hyperparameter search is included.
- Invalid solver/penalty combinations will raise scikit-learn errors.
