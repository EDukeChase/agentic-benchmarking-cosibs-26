# Logistic Regression

Implementation notes:
- Uses `sklearn.linear_model.LogisticRegression` for a regularized binary classifier.
- Exposes common solver, regularization, and class-weight settings through a config dataclass.
- Returns class labels, probabilities, and decision scores.

Assumptions made because the source documentation was incomplete:
- Input features are already numeric and preprocessed.
- The benchmark is binary classification; multiclass support follows scikit-learn defaults but was not the target described in the review.
- Default scikit-learn solver and penalty settings are used unless overridden.

Limitations:
- Convergence is not guaranteed on all datasets without tuning `max_iter`, `solver`, or feature scaling.
- The module does not perform preprocessing, imputation, or calibration.
