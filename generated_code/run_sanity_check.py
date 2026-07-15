"""Run a simple end-to-end sanity check for the logistic regression baseline."""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from config import LogisticRegressionConfig
from training import train_and_evaluate


def main():
    X, y = make_classification(
        n_samples=300,
        n_features=25,
        n_informative=6,
        n_redundant=3,
        random_state=7,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=7, stratify=y
    )
    result = train_and_evaluate(
        X_train,
        y_train,
        X_val,
        y_val,
        config=LogisticRegressionConfig(max_iter=1000),
    )
    print(result["metrics"])


if __name__ == "__main__":
    main()

