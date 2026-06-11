#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm4ckd.metrics import summarize_prediction_frame


def parse_args():
    p = argparse.ArgumentParser(description="Create paper-style balanced accuracy line plot.")
    p.add_argument("--predictions", nargs="+", required=True)
    p.add_argument("--figure", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.concat([pd.read_csv(p) for p in args.predictions], ignore_index=True)
    metrics = summarize_prediction_frame(df)

    x_col = "shots" if "shots" in metrics.columns else "training_size"
    if "model" not in metrics.columns:
        raise ValueError("Prediction metrics need a model column for plotting.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in metrics.groupby("model"):
        gg = g.groupby(x_col, as_index=False)["balanced_accuracy"].mean().sort_values(x_col)
        ax.plot(gg[x_col], gg["balanced_accuracy"], marker="o", label=str(model))
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.45, 1.0)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    out = Path(args.figure)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
