import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

from model import LinearRegressionModel


def _load_ehrshot():
    labels = pd.read_csv('/app/data/EHR_SHOT/labels.csv')
    files = {int(re.search(r'(\d+)', Path(f).stem).group(1)): f for f in Path('/app/data/EHR_SHOT/patient_data_all').glob('*.csv')}
    rows = []
    for _, row in labels.iterrows():
        pid = int(row['patient_id'])
        fp = files.get(pid)
        if fp is None:
            continue
        text = '\n'.join(pd.read_csv(fp)['TEXT'].astype(str).tolist())
        rows.append((pid, text, int(row['new_acutemi'])))
    return pd.DataFrame(rows, columns=['patient_id', 'text', 'y'])


def benchmark():
    df = _load_ehrshot()
    n = len(df)
    train = df.iloc[: int(n * 0.6)]
    test = df.iloc[int(n * 0.8):]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(train['text']).toarray()
    X_test = vectorizer.transform(test['text']).toarray()
    y_train = train['y'].values
    y_test = test['y'].values

    model = LinearRegressionModel()
    model.fit(X_train, y_train)
    probs = model.predict(X_test)
    probs = np.clip(probs, 0.0, 1.0)
    preds = (probs >= 0.5).astype(int)

    return {
        'accuracy': float(accuracy_score(y_test, preds)),
        'f1': float(f1_score(y_test, preds, zero_division=0)),
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall': float(recall_score(y_test, preds, zero_division=0)),
        'auroc': float(roc_auc_score(y_test, probs)),
        'brier': float(brier_score_loss(y_test, probs)),
    }


if __name__ == '__main__':
    print(json.dumps(benchmark(), indent=2))
