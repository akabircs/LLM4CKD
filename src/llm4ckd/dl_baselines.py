"""Tabular DL and foundation-model baselines for LLM4CKD.

This module keeps the registry and shared preprocessing concise, while the
official NODE and SAINT wrappers live in separate modules:

    - llm4ckd.node.OfficialNODEClassifier
    - llm4ckd.saint.OfficialSAINTClassifier

Implemented baselines:
    - TabPFN  : tabpfn package
    - TabNet  : pytorch-tabnet package
    - NODE    : official Qwicen/node ODST wrapper
    - SAINT   : official somepago/saint TabAttention wrapper
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from llm4ckd.ml_baselines import infer_column_types
from llm4ckd.node import OfficialNODEClassifier
from llm4ckd.saint import OfficialSAINTClassifier


def _make_dense_onehot_encoder():
    """Create a dense OneHotEncoder compatible with sklearn versions."""
    from sklearn.preprocessing import OneHotEncoder
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_dl_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Dense preprocessing for tabular DL/foundation baselines."""
    numeric_cols, categorical_cols = infer_column_types(X)
    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _make_dense_onehot_encoder())])
        transformers.append(("cat", categorical_pipe, categorical_cols))
    if not transformers:
        raise ValueError("No usable numeric or categorical columns were found.")
    return ColumnTransformer(transformers, sparse_threshold=0.0)


def _ensure_two_column_proba(proba) -> np.ndarray:
    """Ensure probability output has shape [n_samples, 2]."""
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    if proba.shape[1] == 1:
        p1 = proba[:, 0]
        proba = np.column_stack([1.0 - p1, p1])
    return proba


class TabPFNSklearnClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for TabPFNClassifier."""

    def __init__(self, seed: int = 42, device: str = "cpu"):
        self.seed = seed
        self.device = device

    def fit(self, X, y):
        from tabpfn_client import TabPFNClassifier, set_access_token
        from pathlib import Path
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        tabpfn_api_key = Path("config/tabpfn_api_key.txt").read_text().strip()
        set_access_token(tabpfn_api_key)
        try:
            self.model_ = TabPFNClassifier(device=self.device, random_state=self.seed)
        except TypeError:
            try:
                self.model_ = TabPFNClassifier(device=self.device)
            except TypeError:
                self.model_ = TabPFNClassifier()
        self.model_.fit(X, y)
        self.classes_ = np.array([0, 1], dtype=np.int64)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return _ensure_two_column_proba(self.model_.predict_proba(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class TabNetSklearnClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for pytorch-tabnet TabNetClassifier."""

    def __init__(self, seed: int = 42, max_epochs: int = 200, patience: int = 30, batch_size: int = 32, virtual_batch_size: int = 16, device_name: str = "auto"):
        self.seed = seed
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.device_name = device_name

    def fit(self, X, y):
        from pytorch_tabnet.tab_model import TabNetClassifier
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        effective_batch_size = min(self.batch_size, max(2, len(y)))
        effective_virtual_batch_size = min(self.virtual_batch_size, max(2, effective_batch_size))
        self.model_ = TabNetClassifier(seed=self.seed, verbose=0, device_name=self.device_name)
        self.model_.fit(
            X_train=X,
            y_train=y,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=effective_batch_size,
            virtual_batch_size=effective_virtual_batch_size,
            num_workers=0,
            drop_last=False,
        )
        self.classes_ = np.array([0, 1], dtype=np.int64)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return _ensure_two_column_proba(self.model_.predict_proba(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def dl_model_registry(seed: int = 42, device: str = "cpu", include_optional: bool = True, node_repo_dir: Optional[str] = None, saint_repo_dir: Optional[str] = None) -> Dict[str, BaseEstimator]:
    """Return available DL/foundation baselines."""
    models: Dict[str, BaseEstimator] = {}
    if not include_optional:
        return models
    try:
        import tabpfn  # noqa: F401
        models["TabPFN"] = TabPFNSklearnClassifier(seed=seed, device=device)
    except Exception as exc:
        print(f"[WARN] TabPFN unavailable: {exc}")
    try:
        import pytorch_tabnet  # noqa: F401
        tabnet_device = "cuda" if device == "cuda" else "cpu"
        models["TabNet"] = TabNetSklearnClassifier(seed=seed, device_name=tabnet_device)
    except Exception as exc:
        print(f"[WARN] TabNet unavailable: {exc}")
    try:
        # node_model = OfficialNODEClassifier(seed=seed, device=device, node_repo_dir=node_repo_dir)
        # node_model._import_official_odst()
        # models["NODE"] = node_model
        models["NODE"] = OfficialNODEClassifier(
        seed=seed,
        device=device,
        node_repo_dir=node_repo_dir,
        )
    except Exception as exc:
        print(f"[WARN] NODE unavailable: {exc}")
    try:
        saint_model = OfficialSAINTClassifier(seed=seed, device=device, saint_repo_dir=saint_repo_dir)
        saint_model._import_tabattention()
        models["SAINT"] = saint_model
    except Exception as exc:
        print(f"[WARN] SAINT unavailable: {exc}")
    return models


def make_dl_pipeline(model: BaseEstimator, X_train: pd.DataFrame) -> Pipeline:
    """Create dense preprocessing + DL/foundation model pipeline."""
    return Pipeline([("preprocess", make_dl_preprocessor(X_train)), ("model", model)])


def predict_proba_positive(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities from a fitted estimator."""
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        return _ensure_two_column_proba(proba)[:, 1]
    preds = estimator.predict(X)
    return np.asarray(preds, dtype=float)
