"""Official NODE baseline wrapper for LLM4CKD.

This module uses the official Qwicen/node implementation by importing the
ODST layer from a local clone of:

    https://github.com/Qwicen/node

Expected local clone locations, in search order:
    1. `node_repo_dir` argument
    2. `NODE_REPO_DIR` environment variable
    3. ./external/node
    4. ./node
    5. ~/node

Install/clone example:
    git clone https://github.com/Qwicen/node.git external/node
"""

from __future__ import annotations

import inspect
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _OfficialNODETorchModel(nn.Module):
    """NODE classifier built from official Qwicen/node ODST layers."""

    def __init__(
        self,
        ODST,
        in_features: int,
        num_classes: int = 2,
        num_layers: int = 1,
        num_trees: int = 16,
        depth: int = 2,
        tree_dim: int = 1,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_trees = num_trees
        self.depth = depth
        self.tree_dim = tree_dim
        self.dropout = dropout
        self.device_ = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        choice_function = None
        bin_function = None
        try:
            from lib.nn_utils import entmax15, entmoid15  # type: ignore
            choice_function = entmax15
            bin_function = entmoid15
        except Exception:
            pass

        odst_signature = inspect.signature(ODST.__init__)

        def make_odst(in_f: int) -> nn.Module:
            candidate_kwargs = {
                "in_features": in_f,
                "num_trees": num_trees,
                "depth": depth,
                "tree_dim": tree_dim,
                "choice_function": choice_function,
                "bin_function": bin_function,
                "flatten_output": True,
            }
            kwargs = {}
            for key, value in candidate_kwargs.items():
                if key not in odst_signature.parameters:
                    continue
                if value is None and key in {"choice_function", "bin_function"}:
                    continue
                kwargs[key] = value
            return ODST(**kwargs)

        self.layers = nn.ModuleList()
        current_in = in_features
        total_out = 0
        for _ in range(num_layers):
            layer = make_odst(current_in)
            self.layers.append(layer)
            layer_out = num_trees * tree_dim
            total_out += layer_out
            current_in += layer_out

        self.head = nn.Linear(total_out, num_classes)
        self.to(self.device_)

    @property
    def device(self) -> torch.device:
        return self.device_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        features = x
        outputs = []
        for layer in self.layers:
            if self.dropout > 0:
                features = F.dropout(features, p=self.dropout, training=self.training)
            out = layer(features)
            if out.ndim > 2:
                out = out.reshape(out.shape[0], -1)
            outputs.append(out)
            features = torch.cat([features, out], dim=1)
        all_outputs = torch.cat(outputs, dim=1)
        return self.head(all_outputs)


class OfficialNODEClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible official NODE classifier for LLM4CKD."""

    def __init__(
        self,
        seed: int = 42,
        device: str = "auto",
        node_repo_dir: Optional[str] = None,
        num_layers: int = 1,
        num_trees: int = 16,
        depth: int = 2,
        tree_dim: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 200,
        batch_size: Optional[int] = None,
        dropout: float = 0.0,
        warmup_batch_size: int = 1024,
    ):
        self.seed = seed
        self.device = device
        self.node_repo_dir = node_repo_dir
        self.num_layers = num_layers
        self.num_trees = num_trees
        self.depth = depth
        self.tree_dim = tree_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.warmup_batch_size = warmup_batch_size

    def fit(self, X, y):
        _set_seed(self.seed)
        ODST = self._import_official_odst()
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int64)
        if X_np.ndim != 2:
            raise ValueError(f"NODE expected a 2D feature matrix, got shape {X_np.shape}.")
        if len(X_np) != len(y_np):
            raise ValueError("X and y have inconsistent lengths.")

        self.device_ = self._resolve_device()
        self.n_features_in_ = X_np.shape[1]
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.long)

        self.model_ = _OfficialNODETorchModel(
            ODST=ODST,
            in_features=self.n_features_in_,
            num_classes=2,
            num_layers=self.num_layers,
            num_trees=self.num_trees,
            depth=self.depth,
            tree_dim=self.tree_dim,
            dropout=self.dropout,
            device=self.device_,
        )

        with torch.no_grad():
            warmup_n = min(len(X_tensor), self.warmup_batch_size)
            _ = self.model_(X_tensor[:warmup_n].to(self.device_))

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        n = X_tensor.shape[0]
        effective_batch_size = n if self.batch_size is None else min(self.batch_size, n)
        effective_batch_size = max(1, effective_batch_size)
        indices = np.arange(n)
        self.model_.train()
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, n, effective_batch_size):
                batch_idx = indices[start : start + effective_batch_size]
                xb = X_tensor[batch_idx].to(self.device_)
                yb = y_tensor[batch_idx].to(self.device_)
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
        self.classes_ = np.array([0, 1], dtype=np.int64)
        return self

    def predict_proba(self, X):
        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"NODE expected a 2D feature matrix, got shape {X_np.shape}.")
        if X_np.shape[1] != self.n_features_in_:
            raise ValueError(f"NODE expected {self.n_features_in_} features but got {X_np.shape[1]}.")
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device_)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _candidate_repo_dirs(self) -> list[Path]:
        candidates: list[Path] = []
        if self.node_repo_dir:
            candidates.append(Path(self.node_repo_dir).expanduser())
        env_dir = os.environ.get("NODE_REPO_DIR")
        if env_dir:
            candidates.append(Path(env_dir).expanduser())
        cwd = Path.cwd()
        candidates.extend([cwd / "external" / "node", cwd / "node", cwd / "node_official", Path.home() / "node", Path.home() / "node_official"])
        return candidates

    def _import_official_odst(self):
        for repo_dir in self._candidate_repo_dirs():
            odst_file = repo_dir / "lib" / "odst.py"
            if not odst_file.exists():
                continue
            repo_str = str(repo_dir.resolve())
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)
            try:
                from lib.odst import ODST  # type: ignore
                return ODST
            except Exception as exc:
                raise ImportError(f"Found NODE repository at {repo_dir}, but failed to import lib.odst.ODST.") from exc
        searched = "\n".join(str(p) for p in self._candidate_repo_dirs())
        raise ImportError(
            "Official NODE repository was not found. Clone it first:\n\n"
            "    git clone https://github.com/Qwicen/node.git external/node\n\n"
            "Or set NODE_REPO_DIR to the local NODE repository path.\n\n"
            f"Searched paths:\n{searched}"
        )
