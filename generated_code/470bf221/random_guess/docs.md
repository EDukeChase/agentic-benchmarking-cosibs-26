# Random Guess

Implementation uses `sklearn.dummy.DummyClassifier(strategy="stratified")`.

## Decisions
- Predictions ignore patient features entirely.
- Class sampling follows the observed training distribution.

## Assumptions
- Binary classification benchmarking expects `predict_proba` to be available.

## Limitations
- Performance should approximate the prevalence-weighted no-skill baseline.
- Results vary with `random_state`.
