from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

DISPLAY_NAMES = {
    "age": "Age",
    "gender": "Gender",
    "illiterate": "Illiterate",
    "occupation": "Occupation",
    "marital_status": "Marital status",
    "sleeping_duration": "Sleeping duration",
    "tobacco_smoker": "Tobacco smoker",
    "smokeless_tobacco": "Smokeless tobacco",
    "history_of_hypertension": "History of hypertension",
    "history_of_diabetes": "History of diabetes",
    "heart_disease": "Heart disease",
    "stroke": "Stroke",
    "family_history_of_diabetes": "Family history of diabetes",
    "family_history_of_hypertension": "Family history of hypertension",
    "family_history_of_ckd": "Family history of CKD",
    "body_mass_index": "Body mass index",
    "abdominal_obesity": "Abdominal obesity",
    "undernutrition": "Undernutrition",
    "anemia": "Anemia",
    "presence_of_red_blood_cells_in_urine": "Presence of red blood cells in urine",
    "serum_albumin": "Serum albumin",
    "hypercholesterolemia": "Hypercholesterolemia",
    "hdl_cholesterol": "HDL cholesterol",
    "hypertriglyceridemia": "Hypertriglyceridemia",
    "blood_pressure": "Blood pressure",
    "specific_gravity": "Specific gravity",
    "albumin": "Albumin",
    "sugar": "Sugar",
    "pus_cell": "Pus cell",
    "pus_cell_clumps": "Pus cell clumps",
    "bacteria": "Bacteria",
    "blood_glucose_random": "Blood glucose random",
    "blood_urea": "Blood urea",
    "sodium": "Sodium",
    "potassium": "Potassium",
    "hemoglobin": "Hemoglobin",
    "packed_cell_volume": "Packed cell volume",
    "white_blood_cell_count": "White blood cell count",
    "red_blood_cell_count": "Red blood cell count",
    "coronary_artery_disease": "Coronary artery disease",
    "appetite_condition": "Appetite condition",
    "pedal_edema": "Pedal edema",
}

YES_VALUES = {"yes", "y", "true", "1", "present", "positive", "abnormal", "poor", "good"}
NO_VALUES = {"no", "n", "false", "0", "notpresent", "not_present", "negative", "normal"}


def display_name(feature: str) -> str:
    return DISPLAY_NAMES.get(feature, feature.replace("_", " ").title())


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "?", "nan", "na", "n/a", "none", "null"}:
        return True
    return False


def normalize_value(feature: str, value: Any) -> str:
    """Convert a raw table value into a human-readable prompt value."""
    if _is_missing(value):
        return "Missing"
    if isinstance(value, str):
        s = value.strip().replace("\t", "")
        sl = s.lower().replace(" ", "_").replace("-", "_")
        if sl in YES_VALUES:
            return "Yes"
        if sl in NO_VALUES:
            return "No"
        return re.sub(r"_+", " ", s).strip()
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.4g}"
    return str(value)


def serialize_record_list(row: pd.Series, features: Iterable[str]) -> str:
    """Serialize a patient record as key-value pairs."""
    parts = [f"{display_name(f)}: {normalize_value(f, row.get(f))}" for f in features]
    return ",\n".join(parts)


def _sentence_for_feature(feature: str, value: str) -> str:
    name = display_name(feature).lower()
    v = value
    vl = value.lower()
    if vl == "missing":
        return f"The patient's {name} is missing."
    if vl == "yes":
        return f"The patient has {name}."
    if vl == "no":
        return f"The patient does not have {name}."
    if feature == "gender":
        return f"The patient is {vl}."
    if feature == "age":
        return f"The patient is {v} years old."
    if feature == "body_mass_index":
        # Preserve categories such as obese/overweight if provided; otherwise numeric BMI.
        if re.search(r"[a-zA-Z]", v):
            return f"The patient is {vl} by body mass index."
        return f"The patient's body mass index is {v}."
    if feature == "sleeping_duration":
        return f"The patient sleeps {v} hours per day."
    return f"The patient's {name} is {v}."


def serialize_record_text(row: pd.Series, features: Iterable[str]) -> str:
    """Serialize a patient record as a compact clinical narrative."""
    sentences = []
    for f in features:
        val = normalize_value(f, row.get(f))
        sentences.append(_sentence_for_feature(f, val))
    return " ".join(sentences)


def serialize_record(row: pd.Series, features: Iterable[str], serialization: str) -> str:
    if serialization == "list":
        return serialize_record_list(row, features)
    if serialization == "text":
        return serialize_record_text(row, features)
    raise ValueError("serialization must be 'list' or 'text'")
