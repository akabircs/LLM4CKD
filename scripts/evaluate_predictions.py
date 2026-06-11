#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm4ckd.metrics import summarize_prediction_frame


def parse_args():
    p = argparse.ArgumentParser(description="Compute metrics from a LLM4CKD prediction CSV.")
    p.add_argument("--predictions", nargs="+", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    frames = [pd.read_csv(path) for path in args.predictions]
    df = pd.concat(frames, ignore_index=True)
    metrics = summarize_prediction_frame(df)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out, index=False)
    print(metrics)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
