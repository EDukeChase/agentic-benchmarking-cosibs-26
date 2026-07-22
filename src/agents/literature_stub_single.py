from src.schemas import GeneratedModel

NUMBER_OF_MODELS = 1

response = """
Candidate model for predicting hypertension diagnosis from EHR data (EHRSHOT-format):

1. Logistic Regression (baseline)
   - Task: binary classification (hypertension diagnosis within prediction window)
   - Inputs: structured features engineered from EHR -- most recent systolic/diastolic
     BP readings, BMI, age, sex, smoking status flag, existing diagnosis codes
     (e.g., diabetes, hyperlipidemia as ICD-9/10 codes), medication flags
     (antihypertensives, if present prior to prediction window), family history
     flag if available.
   - Architecture: standard L2-regularized logistic regression (scikit-learn
     LogisticRegression, penalty='l2', solver='lbfgs').
   - Training: standardize continuous features (StandardScaler), fit on train
     split, tune regularization strength C via cross-validation on train split
     (grid over [0.01, 0.1, 1, 10]).
   - Output: predicted probability of hypertension diagnosis; report accuracy,
     F1, and AUROC on the held-out test split.
   - Notes: serves as an interpretable baseline; coefficients directly indicate
     feature contribution.
"""

GENERATED_MODELS = [
    GeneratedModel(
        model_name="logistic_regression",
        resource_name="scikit-learn LogisticRegression documentation",
        resource_link="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
        summary="L2-regularized logistic regression baseline on structured EHR snapshot features.",
        rationale="Interpretable baseline; coefficients directly indicate feature contribution.",
    ),
]
