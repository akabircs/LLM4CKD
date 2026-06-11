from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

TARGET_CANDIDATES = ["ckd", "class", "classification", "target", "label", "y"]

ALIASES = {
    # labels and common UCI aliases
    "bp": "blood_pressure",
    "sg": "specific_gravity",
    "al": "albumin",
    "su": "sugar",
    "rbc": "presence_of_red_blood_cells_in_urine",
    "pc": "pus_cell",
    "pcc": "pus_cell_clumps",
    "ba": "bacteria",
    "bgr": "blood_glucose_random",
    "bu": "blood_urea",
    "sod": "sodium",
    "pot": "potassium",
    "hemo": "hemoglobin",
    "pcv": "packed_cell_volume",
    "wc": "white_blood_cell_count",
    "rc": "red_blood_cell_count",
    "htn": "history_of_hypertension",
    "hypertension": "history_of_hypertension",
    "dm": "history_of_diabetes",
    "diabetes": "history_of_diabetes",
    "cad": "coronary_artery_disease",
    "appet": "appetite_condition",
    "pe": "pedal_edema",
    "ane": "anemia",
    "anaemia": "anemia",
    "sex": "gender",
    "sleep_duration": "sleeping_duration",
    "sleeping_hours": "sleeping_duration",
    "bmi": "body_mass_index",
    "obesity": "body_mass_index",
    "red_blood_cells_in_urine": "presence_of_red_blood_cells_in_urine",
    "urine_rbc": "presence_of_red_blood_cells_in_urine",
    "family_hypertension": "family_history_of_hypertension",
    "fh_hypertension": "family_history_of_hypertension",
    "family_diabetes": "family_history_of_diabetes",
    "fh_diabetes": "family_history_of_diabetes",
    "family_ckd": "family_history_of_ckd",
    "fh_ckd": "family_history_of_ckd",
}


def snake_case(name: str) -> str:
    """Normalize a column name to lowercase snake_case and apply known aliases."""
    s = str(name).strip().replace("\ufeff", "")
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return ALIASES.get(s, s)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [snake_case(c) for c in out.columns]
    return out


def find_target_column(df: pd.DataFrame) -> str:
    candidates = [snake_case(c) for c in TARGET_CANDIDATES]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find a target column. Tried: {TARGET_CANDIDATES}. "
        f"Available columns: {list(df.columns)}"
    )


def encode_target(series: pd.Series, positive_names: Iterable[str], negative_names: Iterable[str]) -> pd.Series:
    pos = {str(x).strip().lower() for x in positive_names}
    neg = {str(x).strip().lower() for x in negative_names}

    def conv(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
            if float(x) == 1.0:
                return 1
            if float(x) == 0.0:
                return 0
        sx = str(x).strip().lower().replace(" ", "_")
        if sx in pos:
            return 1
        if sx in neg:
            return 0
        # UCI often uses "ckd\t" or "notckd"
        sx2 = sx.replace("\t", "")
        if sx2 in pos:
            return 1
        if sx2 in neg:
            return 0
        raise ValueError(f"Unrecognized target value: {x!r}")

    return series.map(conv).astype("float").astype("Int64")


def load_dataset(dataset_name: str, config: Dict, root: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and normalize a dataset from the configured CSV path."""
    if dataset_name not in config["paths"]:
        raise KeyError(f"Unknown dataset {dataset_name!r}; expected one of {list(config['paths'])}")
    path = root / config["paths"][dataset_name]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Place your CSV there or update config/default.yaml."
        )
    df = pd.read_csv(path)
    df = normalize_columns(df)
    target_col = find_target_column(df)
    y = encode_target(df[target_col], config["labels"]["positive_names"], config["labels"]["negative_names"])
    df = df.drop(columns=[target_col])
    df = remove_leakage_features(df, config.get("exclude_features", {}).get(dataset_name, []))
    valid = y.notna()
    return df.loc[valid].reset_index(drop=True), y.loc[valid].astype(int).reset_index(drop=True)


def remove_leakage_features(df: pd.DataFrame, leakage_cols: Iterable[str]) -> pd.DataFrame:
    normalized = {snake_case(c) for c in leakage_cols}
    keep_cols = [c for c in df.columns if snake_case(c) not in normalized]
    return df[keep_cols].copy()


def get_feature_columns(df: pd.DataFrame, dataset_name: str, feature_set: str, config: Dict) -> List[str]:
    """Return configured feature columns that are present in df."""
    feature_sets = config["feature_sets"][dataset_name]
    if feature_set not in feature_sets:
        raise KeyError(f"Unknown feature set {feature_set!r}; expected one of {list(feature_sets)}")
    requested = [snake_case(c) for c in feature_sets[feature_set]]
    present = [c for c in requested if c in df.columns]
    missing = [c for c in requested if c not in df.columns]
    if missing:
        print(f"[WARN] Missing configured columns for {dataset_name}/{feature_set}: {missing}")
    if not present:
        raise ValueError(f"None of the requested features are present. Requested: {requested}")
    return present


def select_features(df: pd.DataFrame, dataset_name: str, feature_set: str, config: Dict) -> pd.DataFrame:
    cols = get_feature_columns(df, dataset_name, feature_set, config)
    return df[cols].copy()


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Convert known missing-value tokens to NaN without imputing."""
    missing_tokens = {"?", "", "na", "n/a", "nan", "none", "null", "missing"}
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(
                lambda x: np.nan if isinstance(x, str) and x.strip().lower() in missing_tokens else x
            )
    return out
