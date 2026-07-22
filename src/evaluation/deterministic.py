"""Repository-owned, deterministic evaluation for the frozen EHRSHOT task."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.settings.config import BenchmarkTaskConfig


# container paths converted into paths that work when code is inspected from project folder
def _real_path(path: str) -> Path:
    # checks whether candidate file exists or is not container style path
    candidate = Path(path)
    if candidate.exists() or not path.startswith("/app/"):
        return candidate
    return Path.cwd() / path.removeprefix("/app/")


# create one row of numeric features for each patient
# builds the machine-learning dataset and returns a pandas feature table, a NumPy array of diagnosis labels, a NumPy array of patient IDs.
def build_features(task: BenchmarkTaskConfig) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    root = _real_path(task.data_root)

    # finished table is cached for later runs (don't have to read a bunch of files every time)
    cache_path = _real_path(
        f"/app/.benchmark_cache/{task.dataset.lower()}_{task.outcome}.pkl"
    )
    if cache_path.exists():
        cached = pd.read_pickle(cache_path)
        return cached["features"], cached["targets"], cached["patient_ids"]

    # labels.csv contains patient IDs and true diagnosis outcomes.
    labels = pd.read_csv(root / "labels.csv")
    # checks if labels.csv contains chosen diagnosis column and configured patient ID column
    if task.outcome not in labels or task.patient_id_column not in labels:
        raise ValueError(f"Task columns missing: {task.patient_id_column}, {task.outcome}")
    # empty list that will hold one feature dictionary per patient
    rows: list[dict[str, float]] = []
    # empty list for true diagnosis values
    targets: list[int] = []
    # empty list for patient ids
    patient_ids: list[int] = []

    # Visit patients in a fixed order so repeated runs build the same table.
    # converts each row into a dictionary
    for record in labels.sort_values(task.patient_id_column).to_dict("records"):
        patient_id = int(record[task.patient_id_column])
        # Constructs the path to that patient’s event file
        patient_file = root / "patient_data_all" / f"patient_{patient_id}.csv"
        if not patient_file.exists():
            continue

        frame = pd.read_csv(patient_file)
        feature_row: dict[str, float] = {"event_count": float(len(frame)), "column_count": float(len(frame.columns))}
        # each row of new dataframe represents one patient's data
        # each column stores means of unique and missing values
        # for non-numerical values, store the amount of unique words and average of their lengths
        # should update soon on preserving meaning for better clinical predictions
        for column in sorted(frame.columns):
            values = frame[column]
            prefix = f"col_{column}"
            feature_row[f"{prefix}_missing"] = float(values.isna().mean())
            feature_row[f"{prefix}_unique"] = float(values.nunique(dropna=True))
            if pd.api.types.is_numeric_dtype(values):
                feature_row[f"{prefix}_mean"] = float(values.mean()) if values.notna().any() else 0.0
                feature_row[f"{prefix}_std"] = float(values.std(ddof=0)) if values.notna().any() else 0.0
            else:
                lengths = values.fillna("").astype(str).str.len()
                feature_row[f"{prefix}_length_mean"] = float(lengths.mean())
        rows.append(feature_row)
        targets.append(int(bool(record[task.outcome])))
        patient_ids.append(patient_id)
    # Models need a rectangular numeric table with no missing or infinite values.
    features = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result = (features, np.asarray(targets), np.asarray(patient_ids))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"features": result[0], "targets": result[1], "patient_ids": result[2]}, cache_path)
    return result


def create_splits(patient_ids: np.ndarray, y: np.ndarray,
                  task: BenchmarkTaskConfig) -> dict[str, list[int]]:
    # Create the split in memory. The fixed random seed still gives the same split on each run.
    train_val, test = train_test_split(
        patient_ids,
        test_size=task.test_fraction,
        random_state=task.seed,
        #  ensures that training and testing subsets maintain the exact same percentage of class labels as your original dataset
        stratify=y,
    )

    # validation_fraction describes a fraction of the full dataset. Convert it
    # into the fraction needed after the test patients have been removed
    validation_share = task.validation_fraction / (1.0 - task.test_fraction)
    # Creates a lookup dictionary connecting each patient ID to its diagnosis label.
    y_by_patient = dict(zip(patient_ids.tolist(), y.tolist()))

    train, validation = train_test_split(
        train_val,
        test_size=validation_share,
        random_state=task.seed,
        stratify=[y_by_patient[int(patient_id)] for patient_id in train_val],
    )

    splits = {
        "train": sorted(map(int, train)),
        "validation": sorted(map(int, validation)),
        "test": sorted(map(int, test)),
    }

    # Make sure patients do not overlap between the three groups.
    expected = set(map(int, patient_ids))
    groups = {name: set(map(int, splits[name])) for name in ("train", "validation", "test")}
    if any(groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Patient leakage detected: data splits overlap")
    if set().union(*groups.values()) != expected:
        raise ValueError("Data splits do not match the available cohort")
    return splits


# Import a model.py file created by the programming agent and return a usable
# model object. Generated modules are given unique names to avoid import clashes.
def _load_model(model_file: Path, seed: int):
    # Generated model files aren't installed packages, so they're imported
    # directly from disk. The hash keeps module names unique across runs that
    # each define their own model.py.
    module_name = f"generated_{model_file.parent.name}_{abs(hash(model_file))}"
    # Creates instructions describing how Python should import the file
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {model_file}")
    module = importlib.util.module_from_spec(spec)
    # Executes the generated model.py file and fills the module with its classes and functions.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    model_type = _resolve_model_type(module, module_name, model_file)
    model = _instantiate_model(module, model_type, seed)
    return model


# Older generated files may use SVC(probability=True). Replace that deprecated
# estimator with scikit-learn's supported probability calibration wrapper.
# def replace_deprecated_svc(model):
#     """Replace SVC(probability=True) with sklearn's supported calibrator."""
#     # SVC's built-in probability=True uses an internal 5-fold CV that sklearn
#     # discourages relying on; CalibratedClassifierCV is the supported way to
#     # get calibrated probabilities out of an SVC.
#     if isinstance(model, SVC):
#         return _calibrated_svc(model)

#     # Generated wrappers usually store the sklearn model in one of these fields.
#     for attribute in ("model", "estimator"):
#         inner_model = getattr(model, attribute, None)
#         if isinstance(inner_model, SVC) and inner_model.probability is True:
#             setattr(model, attribute, _calibrated_svc(inner_model))
#             return model

#     return model


# # Copy the old SVC settings into a new SVC and wrap it with a calibrator that
# # provides predict_proba without using the deprecated probability parameter.
# def _calibrated_svc(old_model: SVC) -> CalibratedClassifierCV:
#     parameters = old_model.get_params()
#     parameters.pop("probability", None)
#     base_model = SVC(**parameters)
#     return CalibratedClassifierCV(base_model, ensemble=False)


# Generated files should expose a class named Model. As a beginner-friendly
# fallback, search for one local class that has fit and prediction methods.
def _resolve_model_type(module, module_name: str, model_file: Path):
    """Find the main model class in a generated Python file."""
    # An explicit `Model = ...` alias is unambiguous; prefer it over guessing.
    if hasattr(module, "Model") and inspect.isclass(module.Model):
        return module.Model

    candidates = []
    for name, value in vars(module).items():
        if not inspect.isclass(value):
            continue
        if value.__module__ != module_name:
            continue

        has_fit = hasattr(value, "fit")
        has_prediction = hasattr(value, "predict") or hasattr(value, "predict_proba")
        if has_fit and has_prediction:
            candidates.append((name, value))

    # If several fit/predict-capable classes exist (e.g. a helper class plus
    # the real model), a "...Model"-suffixed name is the best tiebreaker.
    model_classes = []
    for name, value in candidates:
        if name.lower().endswith("model"):
            model_classes.append(value)

    if len(model_classes) == 1:
        return model_classes[0]
    if len(candidates) == 1:
        return candidates[0][1]

    names = [name for name, _ in candidates]
    raise TypeError(
        f"Could not identify one generated model class in {model_file}; "
        f"compatible candidates were {names}. Define `Model = YourModelClass`."
    )


# Construct the generated model. Some generated classes accept random_state
# directly, while others put it inside a separate Config object.
def _instantiate_model(module, model_type, seed: int):
    """Create the model and provide the random seed when possible."""
    parameters = inspect.signature(model_type).parameters
    if "random_state" in parameters:
        return model_type(random_state=seed)

    # Some generated models take a config object instead of kwargs directly;
    # thread the seed through it if there's exactly one config class to use.
    if "config" in parameters:
        config_candidates = []
        for name, value in vars(module).items():
            is_local_class = inspect.isclass(value) and value.__module__ == module.__name__
            if is_local_class and name.lower().endswith("config"):
                config_candidates.append(value)

        if len(config_candidates) == 1:
            config_type = config_candidates[0]
            config_parameters = inspect.signature(config_type).parameters
            kwargs = {}
            if "random_state" in config_parameters:
                kwargs["random_state"] = seed
            return model_type(config=config_type(**kwargs))

    return model_type()


# Return one positive-class probability for every patient. Regression-like
# models may only provide predict, so their scores are limited to the 0-1 range.
def _probabilities(model, X: np.ndarray) -> np.ndarray:
    # Prefer real probabilities; fall back to treating predict() output as a
    # score for models that don't implement predict_proba.
    if hasattr(model, "predict_proba"):
        values = np.asarray(model.predict_proba(X))
        return values[:, 1] if values.ndim == 2 else values
    values = np.asarray(model.predict(X), dtype=float)
    return np.clip(values, 0.0, 1.0)


# Find the validation-set threshold with the best F1 score. The test labels are
# not used here, which keeps the final test evaluation fair.
def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose a classification threshold using validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5

    # precision and recall contain one extra value that has no matching threshold.
    precision = precision[:-1]
    recall = recall[:-1]
    denominator = precision + recall
    f1_values = np.zeros_like(denominator)
    valid = denominator > 0
    f1_values[valid] = 2 * precision[valid] * recall[valid] / denominator[valid]

    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index])


# Run the same evaluation process for every generated model in one pipeline run.
def evaluate_run(run_id: str, task: BenchmarkTaskConfig) -> dict[str, dict[str, float]]:
    run_dir = _real_path(f"/app/generated_code/{run_id}")
    X, y, patient_ids = build_features(task)
    splits = create_splits(patient_ids, y, task)
    patient_to_row = {}
    for row_number, patient_id in enumerate(patient_ids.tolist()):
        patient_to_row[patient_id] = row_number

    train_idx = [patient_to_row[patient_id] for patient_id in splits["train"]]
    validation_idx = [patient_to_row[patient_id] for patient_id in splits["validation"]]
    test_idx = [patient_to_row[patient_id] for patient_id in splits["test"]]
    # Scaler is fit on train only, then applied to validation/test, so no
    # information about those patients leaks into feature scaling.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[train_idx])
    X_validation = scaler.transform(X.iloc[validation_idx])
    X_test = scaler.transform(X.iloc[test_idx])
    y_train = y[train_idx]
    y_validation = y[validation_idx]
    y_test = y[test_idx]
    results: dict[str, dict[str, float]] = {}
    predictions: dict[str, pd.DataFrame] = {}
    model_dirs = []
    for path in run_dir.iterdir():
        if path.is_dir() and (path / "model.py").exists():
            model_dirs.append(path)

    # Train each model, choose its threshold on validation data, and calculate
    # final metrics using the untouched test patients.
    for model_dir in sorted(model_dirs):
        model = _load_model(model_dir / "model.py", task.seed)
        model.fit(X_train, y_train)
        # Threshold is tuned on validation, then applied to test — the test
        # set is only ever touched for the final probability/metric numbers.
        validation_probability = np.clip(
            _probabilities(model, X_validation), 0.0, 1.0
        )
        threshold = choose_threshold(y_validation, validation_probability)
        probability = np.clip(_probabilities(model, X_test), 0.0, 1.0)
        predicted = (probability >= threshold).astype(int)
        results[model_dir.name] = {
            # Primary metrics for this imbalanced diagnosis task.
            "f1": float(f1_score(y_test, predicted, zero_division=0)),
            "recall": float(recall_score(y_test, predicted, zero_division=0)),
            "precision": float(precision_score(y_test, predicted, zero_division=0)),
            "auroc": float(roc_auc_score(y_test, probability)),
            "brier": float(brier_score_loss(y_test, probability)),
            # Accuracy is secondary because most patients are in the negative class.
            "accuracy": float(accuracy_score(y_test, predicted)),
            "threshold": threshold,
        }
        predictions[model_dir.name] = pd.DataFrame(
            [
                {
                    "patient_id": int(pid),
                    "true_diagnosis": int(truth),
                    "probability": float(prob),
                    "threshold": threshold,
                    "generated_diagnosis": bool(pred),
                }
                for pid, truth, prob, pred in zip(patient_ids[test_idx], y_test, probability, predicted)
            ]
        )
        # Generated code shouldn't own its own test/scoring logic, so each
        # model dir gets a stub that just points back at this module.
        test_file = model_dir / f"test_{model_dir.name}_benchmark.py"
        test_file.write_text(
            "\"\"\"Generated audit pointer; evaluation is owned by src.evaluation.deterministic.\"\"\"\n"
            "from src.evaluation.deterministic import evaluate_run\n"
        )
    # These JSON files are the machine-readable outputs used by later stages.
    (run_dir / "benchmark_results.json").write_text(json.dumps(results, indent=2))
    # predictions holds DataFrames for in-memory use; convert to records only
    # for the JSON file on disk.
    (run_dir / "predictions.json").write_text(
        json.dumps({name: df.to_dict(orient="records") for name, df in predictions.items()}, indent=2)
    )

    # Save model-level metrics in one row per model.
    metrics_rows = []
    for model_name, metrics in results.items():
        metrics_rows.append({"model_name": model_name, **metrics})
    pd.DataFrame(metrics_rows).to_csv(
        run_dir / "benchmark_results.csv", index=False
    )

    # Combine patient predictions from all models into one CSV file.
    prediction_tables = []
    for model_name, prediction_table in predictions.items():
        table = prediction_table.copy()
        table.insert(0, "model_name", model_name)
        prediction_tables.append(table)
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        run_dir / "predictions.csv", index=False
    )
    return results
