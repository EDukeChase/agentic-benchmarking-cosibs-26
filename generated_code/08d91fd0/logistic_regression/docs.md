# Logistic Regression

Implementation: scikit-learn `LogisticRegression` with a configurable regularization strength and class weighting.

Assumptions:
- Binary classification is assumed for benchmarking.
- Inputs are already numerically encoded and preprocessed as needed.

Limitations:
- Sensitive to feature scaling and strong collinearity.
- The default solver may need adjustment for very high-dimensional sparse data.
