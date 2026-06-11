from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_indices(y, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )
    return train_idx, test_idx


def sample_low_data_indices(y_train, n_samples: int, seed: int) -> np.ndarray:
    """Stratified low-data sample from the training partition.

    For very small n, this samples as evenly as possible while respecting class availability.
    """
    y_train = np.asarray(y_train)
    classes, counts = np.unique(y_train, return_counts=True)
    rng = np.random.default_rng(seed)
    selected = []
    base = n_samples // len(classes)
    remainder = n_samples % len(classes)
    for i, cls in enumerate(classes):
        cls_idx = np.where(y_train == cls)[0]
        take = base + (1 if i < remainder else 0)
        take = min(take, len(cls_idx))
        selected.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    if len(selected) < n_samples:
        remaining = np.setdiff1d(np.arange(len(y_train)), np.array(selected), assume_unique=False)
        add = min(n_samples - len(selected), len(remaining))
        selected.extend(rng.choice(remaining, size=add, replace=False).tolist())
    selected = np.array(selected, dtype=int)
    rng.shuffle(selected)
    return selected
