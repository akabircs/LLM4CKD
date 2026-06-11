from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

OUT = Path("data/dataset2.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

ckd = fetch_ucirepo(id=336)

X = ckd.data.features.copy()
y = ckd.data.targets.copy()

df = pd.concat([X, y], axis=1)
df.to_csv(OUT, index=False)

print(f"Saved {OUT}")
print(f"Shape: {df.shape}")
print("Columns:")
print(list(df.columns))
