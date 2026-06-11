from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask] == (y_prob[mask] >= 0.5).astype(int))
        conf = np.mean(np.maximum(y_prob[mask], 1.0 - y_prob[mask]))
        ece += np.mean(mask) * abs(acc - conf)
    return float(ece)


def sensitivity_specificity(y_true, y_pred) -> tuple[float, float]:
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return float(sensitivity), float(specificity)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    sens, spec = sensitivity_specificity(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = np.nan
    return {
        "n": int(len(y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auroc": float(auroc) if not np.isnan(auroc) else np.nan,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "sensitivity": sens,
        "specificity": spec,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
    }


def summarize_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"y_true", "y_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")
    group_cols = [c for c in ["dataset", "feature_set", "model", "family", "seed", "shots", "training_size", "prompt_style", "serialization"] if c in df.columns]
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False) if group_cols else [((), df)]:
        m = compute_metrics(g["y_true"].values, g["y_prob"].values)
        if group_cols:
            if not isinstance(keys, tuple):
                keys = (keys,)
            m.update(dict(zip(group_cols, keys)))
        rows.append(m)
    cols = group_cols + ["n", "balanced_accuracy", "auroc", "macro_f1", "sensitivity", "specificity", "brier", "ece"]
    return pd.DataFrame(rows)[cols]
