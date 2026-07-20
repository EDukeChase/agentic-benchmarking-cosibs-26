"""Sanity tests for logistic regression baseline implementation."""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from config import LogisticRegressionConfig
from experiments.training import predict, train_and_evaluate, train_model


def main():
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    config = LogisticRegressionConfig(max_iter=500)
    model = train_model(X_train, y_train, config=config)
    y_pred, y_proba = predict(model, X_val)
    assert y_pred.shape == y_val.shape
    assert y_proba is not None and y_proba.shape == y_val.shape
    results = train_and_evaluate(X_train, y_train, X_val, y_val, config=config)
    assert "metrics" in results and "model" in results
    assert 0.0 <= results["metrics"]["accuracy"] <= 1.0
    assert 0.0 <= results["metrics"]["f1"] <= 1.0
    print("Sanity checks passed.")


if __name__ == "__main__":
    main()

