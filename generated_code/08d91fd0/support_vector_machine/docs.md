# Support Vector Machine

Implementation: scikit-learn `SVC` with probability estimates enabled by default.

Assumptions:
- Binary classification is assumed.
- Inputs should be scaled/preprocessed upstream for best performance.

Limitations:
- Probability calibration adds training cost.
- Kernel choice and scaling strongly affect performance.
