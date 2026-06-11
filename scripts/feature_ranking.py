#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm4ckd.config import load_config, project_root_from_config
from llm4ckd.data import clean_missing_values, load_dataset, select_features
from llm4ckd.ml_baselines import make_preprocessor


def parse_args():
    p = argparse.ArgumentParser(description="Compute ML feature rankings with SHAP or permutation fallback.")
    p.add_argument("--dataset", choices=["dataset1", "dataset2"], required=True)
    p.add_argument("--feature-set", choices=["all", "selected"], default="selected")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--model", choices=["mlp"], default="mlp")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    root = project_root_from_config(config_path)

    X_raw, y = load_dataset(args.dataset, config, root)
    X = clean_missing_values(select_features(X_raw, args.dataset, args.feature_set, config))
    pre = make_preprocessor(X)
    Xt = pre.fit_transform(X)
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, early_stopping=True, random_state=42)
    model.fit(Xt, y)

    feature_names = list(X.columns)
    try:
        import shap

        # KernelExplainer is slow but generic for MLP; sample background for tractability.
        Xt_dense = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        rng = np.random.default_rng(42)
        bg_idx = rng.choice(np.arange(Xt_dense.shape[0]), size=min(50, Xt_dense.shape[0]), replace=False)
        explainer = shap.KernelExplainer(lambda z: model.predict_proba(z)[:, 1], Xt_dense[bg_idx])
        shap_values = explainer.shap_values(Xt_dense, nsamples=100)
        transformed_names = pre.get_feature_names_out()
        importances = pd.Series(np.abs(shap_values).mean(axis=0), index=transformed_names)
        # Collapse one-hot columns back to original feature names.
        collapsed = {}
        for name, val in importances.items():
            original = name.split("__", 1)[-1]
            original = original.split("_", 1)[0] if name.startswith("cat__") else original
            # robust fallback: match longest source feature contained in transformed name
            matches = [f for f in feature_names if f in name]
            original = max(matches, key=len) if matches else original
            collapsed[original] = collapsed.get(original, 0.0) + float(val)
        ranking = pd.DataFrame({"feature": list(collapsed.keys()), "importance": list(collapsed.values()), "method": "shap"})
    except Exception as exc:
        print(f"[WARN] SHAP failed ({exc}); using absolute first-layer MLP weights as fallback.")
        transformed_names = pre.get_feature_names_out()
        weights = np.abs(model.coefs_[0]).mean(axis=1)
        importances = pd.Series(weights, index=transformed_names)
        collapsed = {}
        for name, val in importances.items():
            matches = [f for f in feature_names if f in name]
            original = max(matches, key=len) if matches else name
            collapsed[original] = collapsed.get(original, 0.0) + float(val)
        ranking = pd.DataFrame({"feature": list(collapsed.keys()), "importance": list(collapsed.values()), "method": "mlp_weight_fallback"})

    ranking = ranking.sort_values("importance", ascending=False).reset_index(drop=True)
    ranking.insert(1, "rank", np.arange(1, len(ranking) + 1))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out, index=False)
    print(ranking)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
