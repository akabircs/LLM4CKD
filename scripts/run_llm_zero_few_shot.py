#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm4ckd.config import load_config, project_root_from_config
from llm4ckd.data import clean_missing_values, load_dataset, select_features
from llm4ckd.llm_inference import make_llm_backend
from llm4ckd.prompts import build_prompt, messages_to_llama_chat_text
from llm4ckd.splits import sample_low_data_indices, stratified_indices


def parse_args():
    p = argparse.ArgumentParser(description="Run zero/few-shot LLM inference or generate prompts.")
    p.add_argument("--dataset", choices=["dataset1", "dataset2"], required=True)
    p.add_argument("--feature-set", choices=["all", "selected"], default="selected")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--prompt-style", choices=["instruction", "chat"], default="instruction")
    p.add_argument("--serialization", choices=["list", "text"], default="list")
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--backend", choices=["prompt_only", "hf", "openai"], default="prompt_only")
    p.add_argument("--model-id", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=None, help="Optional cap for debugging.")
    return p.parse_args()


def render_for_csv(prompt_obj):
    if isinstance(prompt_obj, list):
        return messages_to_llama_chat_text(prompt_obj)
    return prompt_obj


def main():
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    root = project_root_from_config(config_path)

    X_raw, y = load_dataset(args.dataset, config, root)
    X = clean_missing_values(select_features(X_raw, args.dataset, args.feature_set, config))
    features = list(X.columns)
    clf = make_llm_backend(args.backend, model_id=args.model_id)
    model_name = args.model_id or args.backend

    rows = []
    for seed in config["seeds"]:
        if args.shots == 0:
            eval_indices = list(range(len(X)))
            examples = []
        else:
            train_idx, test_idx = stratified_indices(y, test_size=config["test_size"], seed=seed)
            low_idx_rel = sample_low_data_indices(y.iloc[train_idx].values, n_samples=args.shots, seed=seed)
            example_abs_idx = train_idx[low_idx_rel]
            examples = [(X.iloc[i], int(y.iloc[i])) for i in example_abs_idx]
            eval_indices = list(test_idx)

        if args.limit is not None:
            eval_indices = eval_indices[: args.limit]

        for abs_i in tqdm(eval_indices, desc=f"seed={seed}, shots={args.shots}"):
            prompt_obj = build_prompt(
                query_row=X.iloc[abs_i],
                features=features,
                serialization=args.serialization,
                prompt_style=args.prompt_style,
                config=config,
                examples=examples,
            )
            p0, p1 = clf.predict_proba(prompt_obj)
            rows.append({
                "dataset": args.dataset,
                "feature_set": args.feature_set,
                "family": "LLM",
                "model": model_name,
                "seed": seed,
                "shots": args.shots,
                "training_size": args.shots,
                "prompt_style": args.prompt_style,
                "serialization": args.serialization,
                "sample_id": int(abs_i),
                "y_true": int(y.iloc[abs_i]),
                "y_prob": float(p1),
                "y_pred": int(p1 >= 0.5),
                "prompt_tokens": getattr(clf, "last_prompt_tokens", None),
                "prompt": render_for_csv(prompt_obj),
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
