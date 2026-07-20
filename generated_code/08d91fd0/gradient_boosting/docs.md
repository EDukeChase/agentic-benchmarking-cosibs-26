# Gradient Boosting

Implementation: scikit-learn `HistGradientBoostingClassifier`.

Assumptions:
- Tabular numeric inputs are expected.
- Missing values can be handled by the underlying estimator.

Limitations:
- Categorical handling is not explicitly configured here.
- Monotonic/interaction constraints are left unset because feature semantics are unknown.
