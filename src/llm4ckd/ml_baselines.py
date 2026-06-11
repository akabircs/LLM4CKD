from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def infer_column_types(X: pd.DataFrame) -> Tuple[list[str], list[str]]:
    numeric_cols = []
    categorical_cols = []
    for col in X.columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        # Treat as numeric if at least 80% of non-missing values parse as numeric.
        non_missing = X[col].notna().sum()
        numeric_fraction = converted.notna().sum() / max(non_missing, 1)
        if numeric_fraction >= 0.8:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols, categorical_cols = infer_column_types(X)
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def model_registry(seed: int = 42, use_class_weight: bool = True) -> Dict[str, BaseEstimator]:
    class_weight = "balanced" if use_class_weight else None
    models: Dict[str, BaseEstimator] = {
        "RF": RandomForestClassifier(random_state=seed, class_weight=class_weight), #n_estimators=300, 
        "GB": GradientBoostingClassifier(random_state=seed),
        "ET": ExtraTreesClassifier(random_state=seed, class_weight=class_weight),
        "LR": LogisticRegression(class_weight=class_weight, random_state=seed), #max_iter=5000, solver="liblinear", 
        "AB": AdaBoostClassifier(random_state=seed),
        "DT": DecisionTreeClassifier(random_state=seed, class_weight=class_weight),
        "MLP": MLPClassifier(random_state=seed), #hidden_layer_sizes=(64, 32), max_iter=2000, early_stopping=True, 
    }
    try:
        from xgboost import XGBClassifier

        models["XGB"] = XGBClassifier(
            class_weight=class_weight, 
            verbosity=0,
            # n_estimators=200,
            # max_depth=3,
            # learning_rate=0.05,
            # subsample=0.9,
            # colsample_bytree=0.9,
            # objective="binary:logistic",
            # eval_metric="logloss",
            random_state=seed,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        models["LGB"] = LGBMClassifier(
            # n_estimators=200,
            # learning_rate=0.05,
            class_weight=class_weight,
            random_state=seed,
            verbose=-1,
        )
    except Exception:
        pass


def make_pipeline(model: BaseEstimator, X_train: pd.DataFrame) -> Pipeline:
    return Pipeline([
        ("preprocess", make_preprocessor(X_train)),
        ("model", model),
    ])


def predict_proba_positive(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.shape[1] == 1:
            return proba[:, 0]
        return proba[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    preds = estimator.predict(X)
    return np.asarray(preds, dtype=float)
