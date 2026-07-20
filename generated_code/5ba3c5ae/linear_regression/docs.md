# Linear Regression

Implementation notes:
- Uses `sklearn.linear_model.LinearRegression` as an ordinary least squares baseline.
- Intended for binary EHRSHOT targets as a linear-probability model.
- Predictions are clipped to `[0, 1]` by default for convenience.

Assumptions made because the source documentation was incomplete:
- Input features are already numeric and encoded.
- No special missing-value handling or scaling is performed here.
- Binary evaluation can be done by thresholding predicted values at 0.5.

Limitations:
- Outputs are not inherently calibrated probabilities.
- Linear regression is not ideal for classification tasks, but serves as a fast baseline.
