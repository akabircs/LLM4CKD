#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm4ckd.config import load_config, project_root_from_config
from llm4ckd.data import clean_missing_values, load_dataset, select_features
from llm4ckd.ml_baselines import make_pipeline, model_registry, predict_proba_positive
from llm4ckd.splits import sample_low_data_indices, stratified_indices


def parse_args():
    p = argparse.ArgumentParser(description="Run low-data ML baselines for LLM4CKD.")
    p.add_argument("--dataset", choices=["dataset1", "dataset2"], required=True)
    p.add_argument("--feature-set", choices=["all", "selected"], default="selected")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--training-sizes", nargs="*", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    root = project_root_from_config(config_path)

    X_raw, y = load_dataset(args.dataset, config, root)
    X = clean_missing_values(select_features(X_raw, args.dataset, args.feature_set, config))
    training_sizes = args.training_sizes or config["ml"]["training_sizes"]

    rows = []
    for seed in config["seeds"]:
        train_idx, test_idx = stratified_indices(y, test_size=config["test_size"], seed=seed)
        X_train_all = X.iloc[train_idx].reset_index(drop=True)
        y_train_all = y.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        for n_train in training_sizes:
            low_idx = sample_low_data_indices(y_train_all.values, n_samples=n_train, seed=seed)
            X_train = X_train_all.iloc[low_idx].reset_index(drop=True)
            y_train = y_train_all.iloc[low_idx].reset_index(drop=True)
            models = model_registry(seed=seed, use_class_weight=config["ml"].get("use_class_weight_balanced", True))

            for model_name, model in tqdm(models.items(), desc=f"seed={seed}, n={n_train}"):
                try:
                    estimator = make_pipeline(model, X_train)
                    estimator.fit(X_train, y_train)
                    probs = predict_proba_positive(estimator, X_test)
                except Exception as exc:
                    print(f"[WARN] Skipping {model_name} seed={seed} n={n_train}: {exc}")
                    continue
                for i, (yt, prob) in enumerate(zip(y_test, probs)):
                    rows.append({
                        "dataset": args.dataset,
                        "feature_set": args.feature_set,
                        "family": "ML",
                        "model": model_name,
                        "seed": seed,
                        "training_size": n_train,
                        "shots": n_train,
                        "sample_id": int(test_idx[i]),
                        "y_true": int(yt),
                        "y_prob": float(prob),
                        "y_pred": int(float(prob) >= 0.5),
                    })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
