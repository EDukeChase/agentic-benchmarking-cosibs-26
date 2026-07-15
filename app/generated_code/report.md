# EHRSHOT Celiac Disease Classification Report

## Executive summary

This benchmark evaluates a single baseline model for binary celiac disease diagnosis from EHRSHOT patient records: standardized logistic regression with L2 regularization. The model is explicitly designed as an interpretable, fast, and reasonably calibrated baseline for structured and engineered EHR features.

### Cross-model metrics at a glance

| Model | Split | Accuracy | Balanced Accuracy | AUROC | AUPRC | F1 | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression (`logreg`) | Validation | 0.9469 | 0.5000 | 0.5862 | 0.0757 | 0.0000 | 0.0000 | 0.0000 | 0.0534 |
| Logistic Regression (`logreg`) | Test | 0.9350 | 0.4957 | 0.5280 | 0.0656 | 0.0000 | 0.0000 | 0.0000 | 0.0639 |

Interpretation at a glance: the model attains high accuracy, but balanced accuracy is approximately chance and F1/precision/recall are all zero on both validation and test. This indicates that the classifier is dominated by the negative class and does not successfully identify positive celiac cases at the default threshold. AUROC is only modestly above 0.5, and AUPRC is low, suggesting limited ranking performance on an imbalanced task.

## Model 1: Logistic Regression (`logreg`)

### Rationale

The literature review selected logistic regression because celiac diagnosis in EHR data is a binary outcome with a mix of structured variables and engineered features. The key motivation is to establish a fast, interpretable, and well-calibrated baseline before moving to more complex methods. The review also specifically requested scikit-learn's `LogisticRegression` with L2 regularization and standardized input features.

Tradeoffs noted in the review are implicit rather than extensive: logistic regression is simple and transparent, but it may underfit nonlinear patterns and interactions present in EHR data.

### Implementation

The implementation follows the review closely and is built as a scikit-learn pipeline to prevent data leakage during preprocessing.

Key model construction excerpt:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(config: LogisticRegressionConfig) -> Pipeline:
    steps = []
    if config.standardize:
        steps.append(("scaler", StandardScaler()))

    clf = LogisticRegression(
        C=config.C,
        max_iter=config.max_iter,
        penalty=config.penalty,
        solver=config.solver,
        random_state=config.random_state,
        class_weight=config.class_weight,
        n_jobs=config.n_jobs,
        tol=config.tol,
        fit_intercept=config.fit_intercept,
        verbose=config.verbose,
    )
    steps.append(("logreg", clf))
    return Pipeline(steps=steps)
```

Implementation details:

- **Standardization** is applied via `StandardScaler()` inside the pipeline, which keeps scaling confined to the training fold and avoids leakage.
- **Classifier**: `sklearn.linear_model.LogisticRegression`
- **Regularization**: L2 penalty (`penalty="l2"`), consistent with the literature review.
- **Default solver**: `lbfgs`
- **Optimization**: `max_iter=1000` to reduce convergence issues.
- **Thresholding**: probability cutoff is `0.5` when converting predicted probabilities to labels.
- **Positive class handling**: the code uses the probability column corresponding to label `1` when available.

Training and evaluation are handled by a small wrapper:

```python
def train_model(X_train, y_train, config=None):
    if config is None:
        config = LogisticRegressionConfig()
    model = build_model(config)
    model.fit(X_train, y_train)
    return model


def predict(model, X, threshold: float = 0.5):
    if hasattr(model, "predict_proba"):
        proba_matrix = model.predict_proba(X)
        classes = getattr(model, "classes_", None)
        if classes is not None and 1 in classes:
            positive_idx = int(np.where(classes == 1)[0][0])
        else:
            positive_idx = 1 if proba_matrix.shape[1] > 1 else 0
        proba = proba_matrix[:, positive_idx]
        pred = (proba >= threshold).astype(int)
        return pred, proba
    pred = model.predict(X)
    return pred, None
```

This confirms the implementation is a conventional probabilistic binary classifier with a fixed default threshold. The configuration defaults are:

```python
C = 1.0
max_iter = 1000
penalty = "l2"
solver = "lbfgs"
random_state = 42
standardize = True
threshold = 0.5
```

### Results

#### Validation metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9469 |
| Balanced accuracy | 0.5000 |
| AUROC | 0.5862 |
| AUPRC | 0.0757 |
| F1 | 0.0000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| Brier score | 0.0534 |

#### Test metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9350 |
| Balanced accuracy | 0.4957 |
| AUROC | 0.5280 |
| AUPRC | 0.0656 |
| F1 | 0.0000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| Brier score | 0.0639 |

#### Interpretation

The high accuracy is not reflective of strong predictive utility here, because the classifier never predicts the positive class at the default threshold, yielding zero precision, recall, and F1. Balanced accuracy near 0.50 indicates performance at chance when averaging sensitivity and specificity. AUROC is only slightly above random on validation and closer to random on test, so the model has limited ability to rank cases above non-cases. The low AUPRC is also consistent with weak positive-class retrieval in an imbalanced setting.

The Brier score is modest, but given the zero-positive predictions and the low AUROC/AUPRC, the overall picture is that this baseline mostly captures class prevalence rather than clinically useful discrimination.

## Limitations and recommended next steps

### Limitations

- Only one model is present, so there is no within-report model comparison beyond validation versus test.
- The benchmarked logistic regression uses a fixed threshold of 0.5; this may be inappropriate under class imbalance and can produce zero positive predictions.
- The report cannot assess feature importance or coefficient directionality because coefficient outputs are not included in the available benchmark artifacts.
- The benchmark summary does not include confidence intervals, calibration plots, or prevalence statistics, which limits statistical interpretation.

### Recommended next steps

- Tune the decision threshold on validation data to improve sensitivity and F1, especially under class imbalance.
- Consider class weighting or resampling if false negatives are clinically costly.
- Report calibration curves and decision-curve-style analyses to complement AUROC/AUPRC.
- Add stronger baselines or nonlinear models if the goal is to improve discriminative performance beyond a linear separator.
- Inspect learned coefficients after training to confirm whether the model is clinically interpretable and aligned with known celiac-related features.
