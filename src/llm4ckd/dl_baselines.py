from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from llm4ckd.ml_baselines import infer_column_types


def _make_dense_onehot_encoder():
    """Create a dense OneHotEncoder compatible with different sklearn versions."""
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_dl_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Dense preprocessing for tabular DL/foundation baselines.

    The output is dense numeric float data, suitable for TabPFN, TabNet,
    NODE, and SAINT-style neural tabular models.
    """
    numeric_cols, categorical_cols = infer_column_types(X)

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_dense_onehot_encoder()),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        sparse_threshold=0.0,
    )


class TabPFNSklearnClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-style wrapper for TabPFNClassifier."""

    def __init__(self, seed: int = 42, device: str = "cpu"):
        self.seed = seed
        self.device = device

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

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
        proba = self.model_.predict_proba(X)
        return _ensure_two_column_proba(proba)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class TabNetSklearnClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-style wrapper for pytorch-tabnet TabNetClassifier."""

    def __init__(
        self,
        seed: int = 42,
        max_epochs: int = 200,
        patience: int = 30,
        batch_size: int = 32,
        virtual_batch_size: int = 16,
        device_name: str = "auto",
    ):
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
        effective_virtual_batch_size = min(
            self.virtual_batch_size,
            max(2, effective_batch_size),
        )

        self.model_ = TabNetClassifier(
            seed=self.seed,
            verbose=0,
            device_name=self.device_name,
        )

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
        proba = self.model_.predict_proba(X)
        return _ensure_two_column_proba(proba)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class PyTorchTabularClassifier(BaseEstimator, ClassifierMixin):
    """Generic wrapper for PyTorch Tabular models such as NODE or SAINT.

    This wrapper uses dense preprocessed features as continuous variables.
    It will work only if the corresponding config class is available in the
    installed pytorch_tabular package.

    Supported model_name values:
        - NODE
        - SAINT, if SAINTConfig is available in the installed package
    """

    def __init__(
        self,
        model_name: str,
        seed: int = 42,
        max_epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ):
        self.model_name = model_name
        self.seed = seed
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit(self, X, y):
        import random

        import numpy as np
        import torch
        from pytorch_tabular import TabularModel
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
        import pytorch_tabular.models as pt_models

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        config_cls = self._resolve_model_config(pt_models)

        columns = [f"x{i}" for i in range(X.shape[1])]
        train_df = pd.DataFrame(X, columns=columns)
        train_df["target"] = y

        data_config = DataConfig(
            target=["target"],
            continuous_cols=columns,
            categorical_cols=[],
        )

        trainer_config = TrainerConfig(
            max_epochs=self.max_epochs,
            batch_size=min(self.batch_size, max(2, len(y))),
            accelerator="auto",
            devices=1,
            early_stopping=None,
            checkpoints=None,
            progress_bar="none",
        )

        optimizer_config = OptimizerConfig()

        model_config = config_cls(
            task="classification",
            learning_rate=self.learning_rate,
        )

        self.model_ = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            verbose=False,
        )

        # For very-low-data experiments, using the training set as validation
        # avoids failure when n_train is only 4 or 8.
        self.model_.fit(train=train_df, validation=train_df)

        self.columns_ = columns
        self.classes_ = np.array([0, 1], dtype=np.int64)
        return self

    def _resolve_model_config(self, pt_models):
        name = self.model_name.upper()

        candidates = {
            "NODE": ["NODEConfig", "NodeConfig"],
            "SAINT": ["SAINTConfig", "SaintConfig"],
        }

        for candidate in candidates.get(name, []):
            if hasattr(pt_models, candidate):
                return getattr(pt_models, candidate)

        available = [x for x in dir(pt_models) if x.endswith("Config")]
        raise ImportError(
            f"{self.model_name} is not available in the installed pytorch_tabular package. "
            f"Available config classes include: {available}"
        )

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        test_df = pd.DataFrame(X, columns=self.columns_)

        pred = self.model_.predict(test_df)

        proba = _extract_pytorch_tabular_positive_probability(pred)
        return np.column_stack([1.0 - proba, proba])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _ensure_two_column_proba(proba) -> np.ndarray:
    """Ensure probability output has shape [n_samples, 2]."""
    proba = np.asarray(proba, dtype=float)

    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])

    if proba.shape[1] == 1:
        p1 = proba[:, 0]
        proba = np.column_stack([1.0 - p1, p1])

    return proba


def _extract_pytorch_tabular_positive_probability(pred: pd.DataFrame) -> np.ndarray:
    """Extract positive-class probabilities from PyTorch Tabular predictions."""
    columns = list(pred.columns)

    preferred_names = [
        "target_1_probability",
        "target_probability_1",
        "1_probability",
        "class_1_probability",
    ]

    for col in preferred_names:
        if col in columns:
            return pred[col].to_numpy(dtype=float)

    probability_cols = [c for c in columns if "probability" in c.lower()]

    if probability_cols:
        # Prefer the last probability column, which is commonly class 1.
        return pred[probability_cols[-1]].to_numpy(dtype=float)

    prediction_cols = [c for c in columns if "prediction" in c.lower()]

    if prediction_cols:
        return pred[prediction_cols[0]].to_numpy(dtype=float)

    raise ValueError(
        "Could not extract probability from PyTorch Tabular prediction output. "
        f"Prediction columns were: {columns}"
    )


def dl_model_registry(
    seed: int = 42,
    device: str = "cpu",
    include_optional: bool = True,
) -> Dict[str, BaseEstimator]:
    """Return available DL/foundation tabular baselines.

    Models are added only when their optional dependencies are importable.
    """
    models: Dict[str, BaseEstimator] = {}

    if include_optional:
        try:
            import tabpfn  # noqa: F401

            models["TabPFN"] = TabPFNSklearnClassifier(seed=seed, device=device)
        except Exception as exc:
            print(f"[WARN] TabPFN unavailable: {exc}")

        try:
            import pytorch_tabnet  # noqa: F401

            models["TabNet"] = TabNetSklearnClassifier(seed=seed)
        except Exception as exc:
            print(f"[WARN] TabNet unavailable: {exc}")

        try:
            import pytorch_tabular  # noqa: F401

            models["NODE"] = PyTorchTabularClassifier(model_name="NODE", seed=seed)
        except Exception as exc:
            print(f"[WARN] NODE unavailable: {exc}")

        try:
            import pytorch_tabular  # noqa: F401

            models["SAINT"] = PyTorchTabularClassifier(model_name="SAINT", seed=seed)
        except Exception as exc:
            print(f"[WARN] SAINT unavailable: {exc}")

    return models


def make_dl_pipeline(model: BaseEstimator, X_train: pd.DataFrame) -> Pipeline:
    """Create a dense preprocessing + DL model pipeline."""
    return Pipeline(
        [
            ("preprocess", make_dl_preprocessor(X_train)),
            ("model", model),
        ]
    )


def predict_proba_positive(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities."""
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        proba = _ensure_two_column_proba(proba)
        return proba[:, 1]

    preds = estimator.predict(X)
    return np.asarray(preds, dtype=float)
