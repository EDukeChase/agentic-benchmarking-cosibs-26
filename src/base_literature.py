"""Local literature-review fixture used when web search is disabled."""

from src.schemas import GeneratedModel, LiteratureReviewResult


BASE_LITERATURE = LiteratureReviewResult(
    candidates=[
        GeneratedModel(
            model_name="Gradient Boosting",
            resource_name="scikit-learn HistGradientBoostingClassifier documentation",
            resource_link="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html",
            summary=(
                "Histogram-based gradient boosting builds decision trees sequentially so each new "
                "tree corrects errors made by the current ensemble. It supports nonlinear decision "
                "boundaries and is efficient on larger tabular datasets."
            ),
            rationale=(
                "Provides a strong tabular-data benchmark capable of modeling complex interactions "
                "among clinical variables while remaining practical to train."
            ),
        ),
        GeneratedModel(
            model_name="Logistic Regression",
            resource_name="scikit-learn LogisticRegression documentation",
            resource_link="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
            summary=(
                "Regularized logistic regression estimates the probability of a binary outcome "
                "using a linear decision function and a logistic link. It supports class weighting "
                "and several regularization and solver choices."
            ),
            rationale=(
                "Offers an interpretable and efficient classification baseline whose coefficients "
                "can help explain which EHR features contribute to a prediction."
            ),
        ),
        GeneratedModel(
            model_name="Random Forest",
            resource_name="scikit-learn RandomForestClassifier documentation",
            resource_link="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
            summary=(
                "A random forest averages predictions from many decision trees trained on bootstrap "
                "samples with randomized feature subsets. It captures nonlinear effects and feature "
                "interactions without requiring feature scaling."
            ),
            rationale=(
                "Serves as a robust nonlinear benchmark for heterogeneous EHR features and provides "
                "feature-importance estimates for exploratory interpretation."
            ),
        ),
        GeneratedModel(
            model_name="Linear Regression",
            resource_name="scikit-learn LinearRegression documentation",
            resource_link="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html",
            summary=(
                "Ordinary least squares linear regression fits coefficients that minimize "
                "residual sum of squares. For a binary outcome it is a simple linear-probability "
                "baseline, although its predictions are not constrained to the interval [0, 1]."
            ),
            rationale=(
                "Provides a fast, interpretable baseline for determining whether simple linear "
                "relationships in engineered EHR features contain predictive signal. Predictions "
                "must be clipped or thresholded when evaluated as a binary classifier."
            ),
        ),
        GeneratedModel(
            model_name="Random Guess",
            resource_name="scikit-learn DummyClassifier documentation",
            resource_link="https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html",
            summary=(
                "A dummy classifier using the stratified strategy generates predictions by randomly "
                "sampling classes according to the class distribution observed in the training data. "
                "It does not use any patient features when making predictions."
            ),
            rationale=(
                "Provides a no-skill baseline for determining whether trained models perform better "
                "than random guessing while accounting for outcome class imbalance. It also helps verify "
                "that the benchmarking pipeline and evaluation metrics behave as expected."
            ),
        ),
        GeneratedModel(
            model_name="Support Vector Machine",
            resource_name="scikit-learn SVC and probability calibration documentation",
            resource_link="https://scikit-learn.org/stable/modules/calibration.html",
            summary=(
                "A support vector classifier finds a maximum-margin decision boundary and can use "
                "kernels to represent nonlinear relationships. For probability estimates with "
                "scikit-learn 1.9 or newer, wrap SVC in CalibratedClassifierCV with ensemble=False "
                "instead of using the deprecated SVC(probability=True) option."
            ),
            rationale=(
                "Adds a margin-based benchmark that can perform well with standardized, high-dimensional "
                "EHR features and complements tree-based and linear approaches. Use "
                "CalibratedClassifierCV(SVC(...), ensemble=False) to provide predict_proba."
            ),
        ),
    ]
)


def load_base_literature(num_models: int | None = None) -> LiteratureReviewResult:
    """Return a copy of all, or the first ``num_models``, local candidates."""
    candidates = BASE_LITERATURE.candidates
    if num_models is not None:
        candidates = candidates[:num_models]
    return LiteratureReviewResult(candidates=[model.model_copy(deep=True) for model in candidates])
