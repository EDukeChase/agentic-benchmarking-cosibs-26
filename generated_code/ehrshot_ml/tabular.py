from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


@dataclass
class TabularModelConfig:
    model_name: str = "xgboost"
    random_state: int = 42


def make_tabular_model(config: TabularModelConfig):
    if config.model_name == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=config.random_state,
            tree_method="hist",
        )
    if config.model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.random_state,
        )
    raise ValueError(f"Unknown tabular model_name: {config.model_name}")
