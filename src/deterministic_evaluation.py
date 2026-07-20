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

from src.config import BenchmarkTaskConfig


def _real_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists() or not path.startswith("/app/"):
        return candidate
    return Path.cwd() / path.removeprefix("/app/")


def build_features(task: BenchmarkTaskConfig) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    root = _real_path(task.data_root)
    cache_path = _real_path(f"/app/experiments/cache/{task.dataset.lower()}_{task.outcome}.pkl")
    if cache_path.exists():
        cached = pd.read_pickle(cache_path)
        return cached["features"], cached["targets"], cached["patient_ids"]
    labels = pd.read_csv(root / "labels.csv")
    if task.outcome not in labels or task.patient_id_column not in labels:
        raise ValueError(f"Frozen task columns missing: {task.patient_id_column}, {task.outcome}")
    rows: list[dict[str, float]] = []
    targets: list[int] = []
    patient_ids: list[int] = []
    for record in labels.sort_values(task.patient_id_column).to_dict("records"):
        patient_id = int(record[task.patient_id_column])
        patient_file = root / "patient_data_all" / f"patient_{patient_id}.csv"
        if not patient_file.exists():
            continue
        frame = pd.read_csv(patient_file)
        feature_row: dict[str, float] = {"event_count": float(len(frame)), "column_count": float(len(frame.columns))}
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
    features = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result = (features, np.asarray(targets), np.asarray(patient_ids))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"features": result[0], "targets": result[1], "patient_ids": result[2]}, cache_path)
    return result


def load_or_create_splits(patient_ids: np.ndarray, y: np.ndarray, task: BenchmarkTaskConfig) -> dict[str, list[int]]:
    split_path = _real_path(task.split_file)
    if split_path.exists():
        splits = json.loads(split_path.read_text())
    else:
        train_val, test = train_test_split(
            patient_ids, test_size=task.test_fraction, random_state=task.seed, stratify=y
        )
        y_lookup = dict(zip(patient_ids.tolist(), y.tolist()))
        validation_share = task.validation_fraction / (1.0 - task.test_fraction)
        train, validation = train_test_split(
            train_val,
            test_size=validation_share,
            random_state=task.seed,
            stratify=[y_lookup[int(pid)] for pid in train_val],
        )
        splits = {name: sorted(map(int, values)) for name, values in (("train", train), ("validation", validation), ("test", test))}
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(splits, indent=2))
    expected = set(map(int, patient_ids))
    groups = {name: set(map(int, splits[name])) for name in ("train", "validation", "test")}
    if any(groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Patient leakage detected: frozen splits overlap")
    if set().union(*groups.values()) != expected:
        raise ValueError("Frozen split file does not match the available cohort")
    return splits


def _load_model(model_file: Path, seed: int):
    module_name = f"generated_{model_file.parent.name}_{abs(hash(model_file))}"
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    model_type = _resolve_model_type(module, module_name, model_file)
    model = _instantiate_model(module, model_type, seed)
    return replace_deprecated_svc(model)


def replace_deprecated_svc(model):
    """Replace SVC(probability=True) with sklearn's supported calibrator."""
    if isinstance(model, SVC):
        return _calibrated_svc(model)

    # Generated wrappers usually store the sklearn model in one of these fields.
    for attribute in ("model", "estimator"):
        inner_model = getattr(model, attribute, None)
        if isinstance(inner_model, SVC) and inner_model.probability is True:
            setattr(model, attribute, _calibrated_svc(inner_model))
            return model

    return model


def _calibrated_svc(old_model: SVC) -> CalibratedClassifierCV:
    parameters = old_model.get_params()
    parameters.pop("probability", None)
    base_model = SVC(**parameters)
    return CalibratedClassifierCV(base_model, ensemble=False)


def _resolve_model_type(module, module_name: str, model_file: Path):
    """Find the main model class in a generated Python file."""
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


def _instantiate_model(module, model_type, seed: int):
    """Create the model and provide the random seed when possible."""
    parameters = inspect.signature(model_type).parameters
    if "random_state" in parameters:
        return model_type(random_state=seed)

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


def _probabilities(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        values = np.asarray(model.predict_proba(X))
        return values[:, 1] if values.ndim == 2 else values
    values = np.asarray(model.predict(X), dtype=float)
    return np.clip(values, 0.0, 1.0)


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


def evaluate_run(run_id: str, task: BenchmarkTaskConfig) -> dict[str, dict[str, float]]:
    run_dir = _real_path(f"/app/generated_code/{run_id}")
    X, y, patient_ids = build_features(task)
    splits = load_or_create_splits(patient_ids, y, task)
    patient_to_row = {}
    for row_number, patient_id in enumerate(patient_ids.tolist()):
        patient_to_row[patient_id] = row_number

    train_idx = [patient_to_row[patient_id] for patient_id in splits["train"]]
    validation_idx = [patient_to_row[patient_id] for patient_id in splits["validation"]]
    test_idx = [patient_to_row[patient_id] for patient_id in splits["test"]]
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

    for model_dir in sorted(model_dirs):
        model = _load_model(model_dir / "model.py", task.seed)
        model.fit(X_train, y_train)
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
                    "generated_diagnosis": bool(pred),
                }
                for pid, truth, prob, pred in zip(patient_ids[test_idx], y_test, probability, predicted)
            ]
        )
        test_file = model_dir / f"test_{model_dir.name}_benchmark.py"
        test_file.write_text(
            "\"\"\"Generated audit pointer; evaluation is owned by src.deterministic_evaluation.\"\"\"\n"
            "from src.deterministic_evaluation import evaluate_run\n"
        )
    (run_dir / "benchmark_results.json").write_text(json.dumps(results, indent=2))
    (run_dir / "predictions.json").write_text(
        json.dumps({name: df.to_dict(orient="records") for name, df in predictions.items()}, indent=2)
    )
    return results
