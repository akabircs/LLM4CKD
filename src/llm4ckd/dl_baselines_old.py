"""Tabular DL and tabular foundation-model baselines for LLM4CKD.

Implemented baselines:
    - TabPFN  : via tabpfn package
    - TabNet  : via pytorch-tabnet
    - NODE    : via pytorch-tabular NODE backend
    - SAINT   : via official somepago/saint TabAttention implementation

Important SAINT note:
    The SAINT implementation here expects the official SAINT repository to be
    available locally. The code searches for it in:

        1. The `saint_repo_dir` argument
        2. The `SAINT_REPO_DIR` environment variable
        3. `external/saint`
        4. `saint`
        5. `~/saint`

    To install the official SAINT repository locally:

        git clone https://github.com/somepago/saint.git external/saint
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from llm4ckd.ml_baselines import infer_column_types


# ---------------------------------------------------------------------
# Shared preprocessing
# ---------------------------------------------------------------------


def _make_dense_onehot_encoder():
    """Create a dense OneHotEncoder compatible with old and new sklearn."""
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_dl_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create dense preprocessing for tabular DL/foundation baselines.

    Numeric columns:
        median imputation + standard scaling

    Categorical columns:
        most-frequent imputation + dense one-hot encoding

    The final matrix is dense numeric float data, suitable for TabPFN,
    TabNet, NODE, and SAINT.
    """
    numeric_cols, categorical_cols = infer_column_types(X)

    transformers = []

    if numeric_cols:
        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_cols))

    if categorical_cols:
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_dense_onehot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No usable numeric or categorical columns were found.")

    return ColumnTransformer(
        transformers,
        sparse_threshold=0.0,
    )


def _ensure_two_column_proba(proba) -> np.ndarray:
    """Ensure probability output has shape [n_samples, 2]."""
    proba = np.asarray(proba, dtype=float)

    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])

    if proba.shape[1] == 1:
        p1 = proba[:, 0]
        proba = np.column_stack([1.0 - p1, p1])

    return proba


# ---------------------------------------------------------------------
# TabPFN
# ---------------------------------------------------------------------


class TabPFNSklearnClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for TabPFNClassifier."""

    def __init__(self, seed: int = 42, device: str = "cpu"):
        self.seed = seed
        self.device = device

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        # Different TabPFN versions expose slightly different constructor args.
        try:
            self.model_ = TabPFNClassifier(
                device=self.device,
                random_state=self.seed,
            )
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


# ---------------------------------------------------------------------
# TabNet
# ---------------------------------------------------------------------


class TabNetSklearnClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for pytorch-tabnet TabNetClassifier."""

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


# ---------------------------------------------------------------------
# NODE through PyTorch Tabular
# ---------------------------------------------------------------------


class NodeSklearnClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible NODE wrapper using PyTorch Tabular.

    This is a practical repository-level NODE backend. For exact reproduction
    with the original NODE repository, export the same train/test splits and
    run the official NODE implementation separately.
    """

    def __init__(
        self,
        seed: int = 42,
        max_epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ):
        self.seed = seed
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit(self, X, y):
        import torch
        from pytorch_tabular import TabularModel
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

        NODEConfig = self._resolve_node_config()

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        columns = [f"x{i}" for i in range(X.shape[1])]
        train_df = pd.DataFrame(X, columns=columns)
        train_df["target"] = y

        data_config = DataConfig(
            target=["target"],
            continuous_cols=columns,
            categorical_cols=[],
        )

        trainer_config = self._make_trainer_config(y)

        optimizer_config = OptimizerConfig()

        model_config = NODEConfig(
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

        # For n <= 32, using the training set as validation avoids
        # failures caused by tiny validation splits.
        self.model_.fit(train=train_df, validation=train_df)

        self.columns_ = columns
        self.classes_ = np.array([0, 1], dtype=np.int64)
        return self

    @staticmethod
    def _resolve_node_config():
        try:
            from pytorch_tabular.models import NODEConfig

            return NODEConfig
        except Exception:
            pass

        try:
            from pytorch_tabular.models.node import NODEConfig

            return NODEConfig
        except Exception as exc:
            raise ImportError(
                "NODEConfig was not found. Install pytorch-tabular or use the "
                "official NODE repository for exact reproduction."
            ) from exc

    def _make_trainer_config(self, y):
        from pytorch_tabular.config import TrainerConfig

        effective_batch_size = min(self.batch_size, max(2, len(y)))

        # PyTorch Tabular constructor arguments vary across versions.
        try:
            return TrainerConfig(
                max_epochs=self.max_epochs,
                batch_size=effective_batch_size,
                accelerator="auto",
                devices=1,
                early_stopping=None,
                checkpoints=None,
                progress_bar="none",
            )
        except TypeError:
            try:
                return TrainerConfig(
                    max_epochs=self.max_epochs,
                    batch_size=effective_batch_size,
                    accelerator="auto",
                    devices=1,
                    early_stopping=None,
                    checkpoints=None,
                )
            except TypeError:
                return TrainerConfig(
                    max_epochs=self.max_epochs,
                    batch_size=effective_batch_size,
                )

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        test_df = pd.DataFrame(X, columns=self.columns_)

        pred = self.model_.predict(test_df)
        p1 = self._extract_positive_probability(pred)

        return np.column_stack([1.0 - p1, p1])

    @staticmethod
    def _extract_positive_probability(pred: pd.DataFrame) -> np.ndarray:
        columns = list(pred.columns)

        preferred_cols = [
            "target_1_probability",
            "target_probability_1",
            "1_probability",
            "class_1_probability",
        ]

        for col in preferred_cols:
            if col in columns:
                return pred[col].to_numpy(dtype=float)

        probability_cols = [c for c in columns if "probability" in c.lower()]

        if probability_cols:
            return pred[probability_cols[-1]].to_numpy(dtype=float)

        prediction_cols = [c for c in columns if "prediction" in c.lower()]

        if prediction_cols:
            return pred[prediction_cols[0]].to_numpy(dtype=float)

        raise ValueError(
            "Could not extract NODE probability output. "
            f"Prediction columns were: {columns}"
        )

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------
# Official SAINT through somepago/saint TabAttention
# ---------------------------------------------------------------------


class OfficialSAINTSklearnClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for official SAINT TabAttention.

    This implementation follows the official SAINT forward contract:

        logits = model(
            x_categ=x_categ,
            x_cont=X,
            x_categ_enc=x_categ_enc,
            x_cont_enc=x_cont_enc,
        )

    For this reproducibility pipeline, all preprocessed tabular features are
    treated as continuous values. Categorical variables are already converted
    into dense one-hot numeric columns by `make_dl_preprocessor()`.
    """

    def __init__(
        self,
        seed: int = 42,
        device: str = "cpu",
        saint_repo_dir: Optional[str] = None,
        dim: int = 16,
        depth: int = 1,
        heads: int = 1,
        dim_head: int = 16,
        mlp_hidden_mults: tuple[int, ...] = (1,),
        attentiontype: str = "colrow",
        num_special_tokens: int = 1,
        lr: float = 5e-4,
        weight_decay: float = 1e-3,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.seed = seed
        self.device = device
        self.saint_repo_dir = saint_repo_dir
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_hidden_mults = mlp_hidden_mults
        self.attentiontype = attentiontype
        self.num_special_tokens = num_special_tokens
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        TabAttention = self._import_tabattention()

        self._set_seeds(torch)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        n, d = X.shape

        self.device_ = torch.device(self.device)
        self.n_features_in_ = d

        self.model_ = TabAttention(
            categories=[],
            num_continuous=d,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dim_head=self.dim_head,
            mlp_hidden_mults=self.mlp_hidden_mults,
            attentiontype=self.attentiontype,
            num_special_tokens=self.num_special_tokens,
            dim_out=2,
        ).to(self.device_)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        criterion = nn.CrossEntropyLoss()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)

        effective_batch_size = (
            n
            if self.batch_size is None
            else min(self.batch_size, max(1, n))
        )

        loader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            drop_last=False,
        )

        epochs = self._resolve_epochs(n)

        self.model_.train()

        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)

                optimizer.zero_grad()

                logits = self._forward_saint(xb)

                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        self.classes_ = np.array([0, 1], dtype=np.int64)
        return self

    def _resolve_epochs(self, n: int) -> int:
        if self.epochs is not None:
            return int(self.epochs)

        # Conservative few-shot schedule.
        if n <= 4:
            return 20
        if n <= 8:
            return 30
        if n <= 16:
            return 50
        return 80

    def _set_seeds(self, torch_module):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch_module.manual_seed(self.seed)

        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(self.seed)

    def _forward_saint(self, x_cont):
        """Run the official SAINT TabAttention forward contract."""
        import torch

        batch_size = x_cont.shape[0]

        # Raw categorical input: empty because all features are continuous
        # after preprocessing / one-hot encoding.
        x_categ = torch.empty(
            (batch_size, 0),
            dtype=torch.long,
            device=x_cont.device,
        )

        # Continuous features must be encoded through official per-feature MLPs.
        x_cont_enc = self._encode_continuous(x_cont)

        # Encoded categorical features: empty but required by forward contract.
        model_dim = getattr(self.model_, "dim", self.dim)

        x_categ_enc = torch.empty(
            (batch_size, 0, model_dim),
            device=x_cont.device,
        )

        logits = self.model_(
            x_categ=x_categ,
            x_cont=x_cont,
            x_categ_enc=x_categ_enc,
            x_cont_enc=x_cont_enc,
        )

        return logits

    def _encode_continuous(self, x_cont):
        """Encode continuous features using official SAINT per-feature MLPs.

        Input:
            x_cont: shape [B, d]

        Output:
            x_cont_enc: shape [B, d, dim]
        """
        import torch

        cont_embeds = []

        for i, mlp in enumerate(self.model_.simple_MLP):
            xi = x_cont[:, i].unsqueeze(-1)
            cont_embeds.append(mlp(xi))

        return torch.stack(cont_embeds, dim=1)

    def predict_proba(self, X):
        import torch

        X = np.asarray(X, dtype=np.float32)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"SAINT expected {self.n_features_in_} features but got {X.shape[1]}."
            )

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32,
            device=self.device_,
        )

        self.model_.eval()

        with torch.no_grad():
            logits = self._forward_saint(X_tensor)
            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def _import_tabattention(self):
        """Import official SAINT TabAttention from a local saint repo."""
        candidate_dirs = []

        if self.saint_repo_dir:
            candidate_dirs.append(Path(self.saint_repo_dir).expanduser())

        env_dir = os.environ.get("SAINT_REPO_DIR")
        if env_dir:
            candidate_dirs.append(Path(env_dir).expanduser())

        cwd = Path.cwd()

        candidate_dirs.extend(
            [
                cwd / "external" / "saint",
                cwd / "saint",
                Path.home() / "saint",
            ]
        )

        for repo_dir in candidate_dirs:
            model_file = repo_dir / "models" / "model.py"

            if model_file.exists():
                repo_str = str(repo_dir.resolve())

                if repo_str not in sys.path:
                    sys.path.insert(0, repo_str)

                try:
                    from models.model import TabAttention

                    return TabAttention
                except Exception as exc:
                    raise ImportError(
                        f"Found SAINT repository at {repo_dir}, but failed to "
                        f"import models.model.TabAttention."
                    ) from exc

        searched = "\n".join(str(p) for p in candidate_dirs)

        raise ImportError(
            "Official SAINT repository was not found. Clone it first:\n\n"
            "    git clone https://github.com/somepago/saint.git external/saint\n\n"
            "Or set the SAINT_REPO_DIR environment variable to the local SAINT "
            "repository path.\n\n"
            f"Searched paths:\n{searched}"
        )


# ---------------------------------------------------------------------
# Registry and pipeline helpers
# ---------------------------------------------------------------------


def dl_model_registry(
    seed: int = 42,
    device: str = "cpu",
    include_optional: bool = True,
    saint_repo_dir: Optional[str] = None,
) -> Dict[str, BaseEstimator]:
    """Return available DL/foundation baselines.

    Models are added only when their optional dependencies are importable.
    Missing optional dependencies are reported as warnings and skipped.
    """
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

        models["TabNet"] = TabNetSklearnClassifier(
            seed=seed,
            device_name=tabnet_device,
        )
    except Exception as exc:
        print(f"[WARN] TabNet unavailable: {exc}")

    try:
        import pytorch_tabular  # noqa: F401

        models["NODE"] = NodeSklearnClassifier(seed=seed)
    except Exception as exc:
        print(f"[WARN] NODE unavailable: {exc}")

    try:
        saint_model = OfficialSAINTSklearnClassifier(
            seed=seed,
            device=device,
            saint_repo_dir=saint_repo_dir,
        )

        # Check importability early so run logs clearly show whether SAINT
        # is available before fitting begins.
        saint_model._import_tabattention()

        models["SAINT"] = saint_model
    except Exception as exc:
        print(f"[WARN] SAINT unavailable: {exc}")

    return models


def make_dl_pipeline(model: BaseEstimator, X_train: pd.DataFrame) -> Pipeline:
    """Create dense preprocessing + DL model pipeline."""
    return Pipeline(
        [
            ("preprocess", make_dl_preprocessor(X_train)),
            ("model", model),
        ]
    )


def predict_proba_positive(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities from a fitted estimator."""
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        proba = _ensure_two_column_proba(proba)
        return proba[:, 1]

    preds = estimator.predict(X)
    return np.asarray(preds, dtype=float)
