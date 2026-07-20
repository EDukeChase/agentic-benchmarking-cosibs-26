# Gradient Boosting

Implementation uses `sklearn.ensemble.HistGradientBoostingClassifier`.

## Decisions
- Chosen to match the literature review's emphasis on efficient tabular boosting.
- Provides `predict_proba` directly.

## Assumptions
- Input is a dense or sparse tabular representation accepted by scikit-learn.

## Limitations
- HistGradientBoostingClassifier has specific constraints around categorical handling and missing values; defaults are left to scikit-learn.
