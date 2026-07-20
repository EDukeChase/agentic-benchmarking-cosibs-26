import numpy as np
from sklearn.datasets import make_classification

from linear_regression.model import LinearRegressionModel
from logistic_regression.model import LogisticRegressionModel
from random_forest.model import RandomForestModel
from gradient_boosting.model import GradientBoostingModel
from support_vector_machine.model import SupportVectorMachineModel

X, y = make_classification(n_samples=200, n_features=20, n_informative=5, random_state=0)
models = [
    LinearRegressionModel(),
    LogisticRegressionModel(max_iter=200),
    RandomForestModel(n_estimators=20, random_state=0),
    GradientBoostingModel(random_state=0, max_iter=20),
    SupportVectorMachineModel(probability=True, random_state=0),
]
for m in models:
    m.fit(X, y)
    metrics = m.evaluate(X, y)
    assert set(metrics).issuperset({"accuracy", "auroc", "auprc"})
    print(type(m).__name__, metrics)
