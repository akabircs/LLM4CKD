"""Official SAINT baseline wrapper for LLM4CKD.

This module uses the official somepago/saint implementation by importing
TabAttention from a local clone of:

    https://github.com/somepago/saint

Expected local clone locations, in search order:
    1. `saint_repo_dir` argument
    2. `SAINT_REPO_DIR` environment variable
    3. ./external/saint
    4. ./saint
    5. ~/saint

Install/clone example:
    git clone https://github.com/somepago/saint.git external/saint
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
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class OfficialSAINTClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for official SAINT TabAttention.

    All preprocessed LLM4CKD features are treated as continuous values. Any
    categorical variables are one-hot encoded by the shared DL preprocessor
    before SAINT receives the dense numeric matrix.
    """

    def __init__(
        self,
        seed: int = 42,
        device: str = "auto",
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
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
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
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout

    def fit(self, X, y):
        _set_seed(self.seed)
        TabAttention = self._import_tabattention()
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int64)
        if X_np.ndim != 2:
            raise ValueError(f"SAINT expected a 2D feature matrix, got shape {X_np.shape}.")
        if len(X_np) != len(y_np):
            raise ValueError("X and y have inconsistent lengths.")

        n, d = X_np.shape
        self.device_ = self._resolve_device()
        self.n_features_in_ = d
        self.model_ = self._build_model(TabAttention, n_continuous=d).to(self.device_)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        effective_batch_size = n if self.batch_size is None else min(self.batch_size, n)
        effective_batch_size = max(1, effective_batch_size)
        loader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, drop_last=False)
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

    def predict_proba(self, X):
        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"SAINT expected a 2D feature matrix, got shape {X_np.shape}.")
        if X_np.shape[1] != self.n_features_in_:
            raise ValueError(f"SAINT expected {self.n_features_in_} features but got {X_np.shape[1]}.")
        X_tensor = torch.tensor(X_np, dtype=torch.float32, device=self.device_)
        self.model_.eval()
        with torch.no_grad():
            logits = self._forward_saint(X_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def _resolve_epochs(self, n: int) -> int:
        if self.epochs is not None:
            return int(self.epochs)
        if n <= 4:
            return 20
        if n <= 8:
            return 30
        if n <= 16:
            return 50
        return 80

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build_model(self, TabAttention, n_continuous: int):
        signature = inspect.signature(TabAttention.__init__)
        candidate_kwargs = {
            "categories": [],
            "num_continuous": n_continuous,
            "dim": self.dim,
            "depth": self.depth,
            "heads": self.heads,
            "dim_head": self.dim_head,
            "mlp_hidden_mults": self.mlp_hidden_mults,
            "attentiontype": self.attentiontype,
            "num_special_tokens": self.num_special_tokens,
            "dim_out": 2,
            "attn_dropout": self.attn_dropout,
            "ff_dropout": self.ff_dropout,
        }
        kwargs = {key: value for key, value in candidate_kwargs.items() if key in signature.parameters}
        return TabAttention(**kwargs)

    def _encode_continuous(self, x_cont: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.model_, "simple_MLP"):
            raise AttributeError(
                "The imported SAINT TabAttention model does not expose simple_MLP. "
                "Please check that you are using the official somepago/saint implementation."
            )
        if len(self.model_.simple_MLP) != x_cont.shape[1]:
            raise ValueError(
                "SAINT simple_MLP count does not match the number of continuous features: "
                f"{len(self.model_.simple_MLP)} vs {x_cont.shape[1]}."
            )
        cont_embeds = []
        for i, mlp in enumerate(self.model_.simple_MLP):
            xi = x_cont[:, i].unsqueeze(-1)
            cont_embeds.append(mlp(xi))
        return torch.stack(cont_embeds, dim=1)

    def _forward_saint(self, x_cont: torch.Tensor) -> torch.Tensor:
        batch_size = x_cont.shape[0]
        x_categ = torch.empty((batch_size, 0), dtype=torch.long, device=x_cont.device)
        x_cont_enc = self._encode_continuous(x_cont)
        model_dim = getattr(self.model_, "dim", self.dim)
        x_categ_enc = torch.empty((batch_size, 0, model_dim), device=x_cont.device)
        logits = self.model_(
            x_categ=x_categ,
            x_cont=x_cont,
            x_categ_enc=x_categ_enc,
            x_cont_enc=x_cont_enc,
        )
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return logits

    def _candidate_repo_dirs(self) -> list[Path]:
        candidates: list[Path] = []
        if self.saint_repo_dir:
            candidates.append(Path(self.saint_repo_dir).expanduser())
        env_dir = os.environ.get("SAINT_REPO_DIR")
        if env_dir:
            candidates.append(Path(env_dir).expanduser())
        cwd = Path.cwd()
        candidates.extend([cwd / "external" / "saint", cwd / "saint", Path.home() / "saint"])
        return candidates

    def _import_tabattention(self):
        for repo_dir in self._candidate_repo_dirs():
            model_file = repo_dir / "models" / "model.py"
            if not model_file.exists():
                continue
            repo_str = str(repo_dir.resolve())
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)
            try:
                from models.model import TabAttention  # type: ignore
                return TabAttention
            except Exception as exc:
                raise ImportError(f"Found SAINT repository at {repo_dir}, but failed to import models.model.TabAttention.") from exc
        searched = "\n".join(str(p) for p in self._candidate_repo_dirs())
        raise ImportError(
            "Official SAINT repository was not found. Clone it first:\n\n"
            "    git clone https://github.com/somepago/saint.git external/saint\n\n"
            "Or set SAINT_REPO_DIR to the local SAINT repository path.\n\n"
            f"Searched paths:\n{searched}"
        )
