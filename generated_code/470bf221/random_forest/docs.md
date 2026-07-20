# Random Forest

Implementation uses `sklearn.ensemble.RandomForestClassifier`.

## Decisions
- Uses a moderately large default number of trees for stability.
- No feature scaling is required.

## Assumptions
- Works on tabular EHR features in the benchmark pipeline.

## Limitations
- Feature importance interpretation is not implemented here.
- Missing-value handling follows scikit-learn's random forest behavior.
