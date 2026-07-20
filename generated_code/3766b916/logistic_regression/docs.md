# Logistic Regression

## Implementation notes
- Uses `sklearn.linear_model.LogisticRegression` as the core classifier.
- Exposes `predict_scores` for positive-class probabilities and `predict` for thresholded labels.
- Provides a small importable API with `fit`, `evaluate`, `get_params`, and `set_params`.

## Assumptions
- Features are already numeric and suitably preprocessed.
- Labels are binary and encoded as 0/1.
- The model is intended for binary classification benchmarking; multinomial use is not the primary target.

## Known limitations
- `log_loss` assumes binary probabilities and will need adjustment for multiclass use.
- Solver/penalty compatibility is delegated to scikit-learn; invalid combinations will raise the underlying sklearn error.
- Coefficients are interpretable, but not causal.
