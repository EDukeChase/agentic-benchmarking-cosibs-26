# Benchmark Report

## Overall Summary

Benchmarking used validation-set threshold selection for each model, and accuracy is reported as a secondary descriptive metric because the outcome is imbalanced. Linear Regression (scikit-learn LinearRegression; threshold 0.13599773701077875) had AUROC 0.5745073891625616, F1 0.0, recall 0.0, precision 0.0, Brier score 0.05592317171620198, and accuracy 0.926829268292683. Despite the highest accuracy, it failed to identify any positives, so it is not a useful classifier in this setting; the documentation also notes that its outputs are clipped linear-probability scores rather than true probabilities. Random Guess (DummyClassifier(strategy="stratified"); threshold 0.0) had AUROC 0.47629310344827586, F1 0.1076923076923077, recall 1.0, precision 0.056910569105691054, Brier score 0.1016260162601626, and accuracy 0.056910569105691054. It serves as a no-skill baseline; its perfect recall reflects the thresholding rule rather than genuine discrimination, and the low AUROC and poor Brier score indicate weak performance. Logistic Regression (threshold 0.07568302111785832) had AUROC 0.5317118226600985, F1 0.13513513513513514, recall 0.35714285714285715, precision 0.08333333333333333, Brier score 0.054990316016338685, and accuracy 0.7398373983739838. This is an interpretable baseline with modest detection of positives but only slightly better than chance discrimination. Random Forest (threshold 0.035) had AUROC 0.5426416256157636, F1 0.13592233009708737, recall 0.5, precision 0.07865168539325842, Brier score 0.06260121951219512, and accuracy 0.6382113821138211. It improved recall over Logistic Regression but remained limited in precision and discrimination. Gradient Boosting (HistGradientBoostingClassifier; threshold 0.0015839891470616433) had the highest AUROC at 0.6074507389162562, with F1 0.08450704225352113, recall 0.21428571428571427, precision 0.05263157894736842, Brier score 0.059768365803562956, and accuracy 0.7357723577235772. Although its AUROC was best among the candidates, its low F1 and precision and only moderate recall indicate limited practical classification performance. Overall, the strongest evidence of signal came from Gradient Boosting by AUROC, while Logistic Regression and Random Forest offered similar low-to-moderate F1; however, none of the models showed strong balanced performance, and the linear model’s high accuracy was not meaningful because recall and F1 were zero.

## Recommendations

Do not select Linear Regression as a classifier despite its high accuracy, because it identified no positive cases. Random Guess should be retained only as a no-skill reference. Among the trained models, Gradient Boosting is the best candidate if the goal is ranking/discrimination, given the highest AUROC, but its low F1 and precision limit its utility as a standalone diagnostic classifier. Logistic Regression and Random Forest provide comparable, modest positive-class detection and may be preferable only if interpretability or simpler implementation is prioritized over marginal metric differences. Any final choice should explicitly consider the validation-set thresholding strategy, because the reported F1, recall, and precision are threshold-dependent and the class imbalance makes accuracy insufficient for model selection.

## Model Results

Primary metrics are shown first. Accuracy is secondary because the outcome is imbalanced.

| Model | AUROC | F1 | Recall | Precision | Brier | Threshold | Accuracy (secondary) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Linear Regression | 0.5745073891625616 | 0.0 | 0.0 | 0.0 | 0.05592317171620198 | 0.13599773701077875 | 0.926829268292683 | success |
| Random Guess | 0.47629310344827586 | 0.1076923076923077 | 1.0 | 0.056910569105691054 | 0.1016260162601626 | 0.0 | 0.056910569105691054 | success |
| Logistic Regression | 0.5317118226600985 | 0.13513513513513514 | 0.35714285714285715 | 0.08333333333333333 | 0.054990316016338685 | 0.07568302111785832 | 0.7398373983739838 | success |
| Random Forest | 0.5426416256157636 | 0.13592233009708737 | 0.5 | 0.07865168539325842 | 0.06260121951219512 | 0.035 | 0.6382113821138211 | success |
| Gradient Boosting | 0.6074507389162562 | 0.08450704225352113 | 0.21428571428571427 | 0.05263157894736842 | 0.059768365803562956 | 0.0015839891470616433 | 0.7357723577235772 | success |

### Linear Regression

#### Rationale

Provides a fast, interpretable baseline for determining whether simple linear relationships in engineered EHR features contain predictive signal. Predictions must be clipped or thresholded when evaluated as a binary classifier.

#### Implementation Notes

# Linear Regression

Implementation uses `sklearn.linear_model.LinearRegression` as a linear-probability baseline for binary EHR outcomes.

## Decisions
- `predict_proba` is synthesized by clipping raw regression outputs to `[0, 1]` and returning `[1-p, p]`.
- `predict` thresholds the raw regression output at `0.5`.

## Assumptions
- Target labels are binary.
- Downstream evaluation can consume a 2-column probability array.

## Limitations
- Outputs are not true probabilities and may be poorly calibrated.
- Predictions outside `[0, 1]` are clipped for benchmarking convenience.


### Random Guess

#### Rationale

Provides a no-skill baseline for determining whether trained models perform better than random guessing while accounting for outcome class imbalance. It also helps verify that the benchmarking pipeline and evaluation metrics behave as expected.

#### Implementation Notes

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


### Logistic Regression

#### Rationale

Offers an interpretable and efficient classification baseline whose coefficients can help explain which EHR features contribute to a prediction.

#### Implementation Notes

# Logistic Regression

Implementation uses `sklearn.linear_model.LogisticRegression`.

## Decisions
- Exposes both `predict` and `predict_proba` for benchmarking.
- Uses a high default `max_iter` to reduce convergence failures.

## Assumptions
- Binary labels are used in the main benchmark setting.

## Limitations
- Solver/regularization details are left configurable via `**kwargs`.


### Random Forest

#### Rationale

Serves as a robust nonlinear benchmark for heterogeneous EHR features and provides feature-importance estimates for exploratory interpretation.

#### Implementation Notes

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


### Gradient Boosting

#### Rationale

Provides a strong tabular-data benchmark capable of modeling complex interactions among clinical variables while remaining practical to train.

#### Implementation Notes

# Gradient Boosting

Implementation uses `sklearn.ensemble.HistGradientBoostingClassifier`.

## Decisions
- Chosen to match the literature review's emphasis on efficient tabular boosting.
- Provides `predict_proba` directly.

## Assumptions
- Input is a dense or sparse tabular representation accepted by scikit-learn.

## Limitations
- HistGradientBoostingClassifier has specific constraints around categorical handling and missing values; defaults are left to scikit-learn.
