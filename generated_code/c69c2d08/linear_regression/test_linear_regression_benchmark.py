from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

from generated_code.c69c2d08.linear_regression.model import LinearRegressionModel


BASE = Path('/app')
MIMIC_DIR = BASE / 'data' / 'MIMIC_tabular'
EHR_DIR = BASE / 'data' / 'EHR_SHOT'


def _build_feature_matrix_mimic():
    diag = pd.read_csv(MIMIC_DIR / 'diagnosis.csv')
    rows = []
    labels = []
    for _, row in diag.iterrows():
        file_name = row['file']
        text_path = MIMIC_DIR / 'inputs' / file_name
        df = pd.read_csv(text_path)
        text = ' '.join(df['TEXT'].fillna('').astype(str).tolist())
        rows.append(text)
        labels.append(1 if 'pneumonia' in str(row['diagnoses']).lower() else 0)
    return pd.DataFrame({'text': rows}), np.asarray(labels)


def _build_feature_matrix_ehr():
    labels = pd.read_csv(EHR_DIR / 'labels.csv')
    rows = []
    y = []
    for _, row in labels.iterrows():
        pid = int(row['patient_id'])
        p = EHR_DIR / 'patient_data_all' / f'patient_{pid}.csv'
        df = pd.read_csv(p)
        text = ' '.join(df['TEXT'].fillna('').astype(str).tolist())
        rows.append(text)
        y.append(int(bool(row['new_acutemi'])))
    return pd.DataFrame({'text': rows}), np.asarray(y)


def _featurize_text(train_text, test_text):
    vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_train = vec.fit_transform(train_text)
    X_test = vec.transform(test_text)
    return X_train.toarray(), X_test.toarray()


def _metrics(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= 0.5).astype(int)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'auroc': float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else float('nan'),
        'brier': float(brier_score_loss(y_true, y_score)),
    }


def benchmark():
    X, y = _build_feature_matrix_mimic()
    split = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    X_train_vec, X_test_vec = _featurize_text(X_train['text'], X_test['text'])
    model = LinearRegressionModel()
    model.fit(X_train_vec, y_train)
    score = np.clip(model.predict(X_test_vec), 0.0, 1.0)
    return _metrics(y_test, score)


if __name__ == '__main__':
    print(json.dumps(benchmark(), indent=2, sort_keys=True))
