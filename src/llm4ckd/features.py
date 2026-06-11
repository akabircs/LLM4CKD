"""Feature serialization utilities for LLM4CKD.

This module converts harmonized CKD tabular records into list-style and
text-style prompt inputs.

Important serialization rule:
    Missing value  -> omit the feature from the prompt
    False / 0 / No -> write an explicit negative statement
    True / 1 / Yes -> write an explicit positive statement
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


DISPLAY_NAMES: dict[str, str] = {
    # Dataset-1 / harmonized features
    "age": "Age",
    "gender": "Gender",
    "gender_Male": "Gender",
    "gender_Female": "Gender",
    "gender_male": "Gender",
    "gender_female": "Gender",
    "illiterate": "Illiterate",
    "occupation": "Occupation",
    "occupation_Farmer": "Occupation",
    "occupation_Housewife": "Occupation",
    "occupation_Others": "Occupation",
    "occupation_farmer": "Occupation",
    "occupation_housewife": "Occupation",
    "occupation_others": "Occupation",
    "marital_status": "Marital status",
    "maritial status_Married": "Marital status",
    "maritial status_Widowed": "Marital status",
    "maritial status_Others": "Marital status",
    "marital status_Married": "Marital status",
    "marital status_Widowed": "Marital status",
    "marital status_Others": "Marital status",
    "sleeping_duration": "Sleeping duration",
    "sleeping less than 7 hours daily": "Sleeping less than 7 hours daily",
    "daily_sleep_less_than_7h": "Sleeping less than 7 hours daily",
    "tobacco_smoker": "Tobacco smoker",
    "tobacco smoker": "Tobacco smoker",
    "smokeless_tobacco": "Smokeless tobacco",
    "smokeless tobacco user": "Smokeless tobacco user",
    "history_of_hypertension": "History of hypertension",
    "has hypertension": "History of hypertension",
    "history_of_diabetes": "History of diabetes",
    "has diabetes": "History of diabetes",
    "heart_disease": "Heart disease",
    "has history of heart disease": "Heart disease",
    "stroke": "Stroke",
    "has history of stroke": "Stroke",
    "family_history_of_diabetes": "Family history of diabetes",
    "has family history of diabetes": "Family history of diabetes",
    "family_history_of_hypertension": "Family history of hypertension",
    "has family history of hypertension": "Family history of hypertension",
    "family_history_of_ckd": "Family history of chronic kidney disease",
    "family_history_of_chronic_kidney_disease": "Family history of chronic kidney disease",
    "has family history of chronic kidney disease": "Family history of chronic kidney disease",
    "body_mass_index": "Body mass index",
    "body mass index_Underweight": "Body mass index",
    "body mass index_Normal": "Body mass index",
    "body mass index_Overweight": "Body mass index",
    "body mass index_Obese": "Body mass index",
    "body_mass_index_underweight": "Body mass index",
    "body_mass_index_normal": "Body mass index",
    "body_mass_index_overweight": "Body mass index",
    "body_mass_index_obese": "Body mass index",
    "abdominal_obesity": "Abdominal obesity",
    "has abdominal obesity": "Abdominal obesity",
    "undernutrition": "Undernutrition",
    "has undernutrition": "Undernutrition",
    "anemia": "Anemia",
    "has anemia": "Anemia",
    "presence_of_red_blood_cells_in_urine": "Presence of red blood cells in urine",
    "has presence of red blood cells in urine": "Presence of red blood cells in urine",
    "serum_albumin": "Serum albumin",
    "low_serum_albumin": "Low serum albumin",
    "has low serum albumin": "Low serum albumin",
    "hypercholesterolemia": "Hypercholesterolemia",
    "has hyper-cholesterolemia": "Hypercholesterolemia",
    "has hypercholesterolemia": "Hypercholesterolemia",
    "hdl_cholesterol": "HDL cholesterol",
    "low_hdl_cholesterol": "Low HDL cholesterol",
    "has low HDL cholesterol": "Low HDL cholesterol",
    "hypertriglyceridemia": "Hypertriglyceridemia",
    "has hypertriglyceridemia": "Hypertriglyceridemia",

    # Dataset-2 / UCI CKD harmonized features
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


def _feature_key(feature: str) -> str:
    """Normalize a feature name for robust matching."""
    return (
        str(feature)
        .strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace("+", "_plus")
        .replace(">", "_greater_than_")
        .replace("<", "_less_than_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def _is_missing(value: Any) -> bool:
    """Return True if a feature value is missing or unknown."""
    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except Exception:
        pass

    text = str(value).strip().lower()
    return text in {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "?",
        "missing",
        "unknown",
        "not available",
        "not_available",
    }


def _as_bool(value: Any) -> bool:
    """Convert common binary encodings to bool for prompt serialization.

    Missing values should be handled before calling this function.
    """
    if isinstance(value, bool):
        return value

    try:
        if isinstance(value, (int, float)) and not pd.isna(value):
            return bool(int(value))
    except Exception:
        pass

    text = str(value).strip().lower()

    if text in {
        "1",
        "true",
        "yes",
        "y",
        "present",
        "positive",
        "abnormal",
        "ckd",
    }:
        return True

    if text in {
        "0",
        "false",
        "no",
        "n",
        "notpresent",
        "not present",
        "negative",
        "normal",
        "notckd",
        "not ckd",
        "non-ckd",
        "non_ckd",
    }:
        return False

    return False


def normalize_value(feature: str, value: Any) -> str:
    """Normalize a raw feature value for list-style serialization."""
    if _is_missing(value):
        return ""

    if isinstance(value, bool):
        return "Yes" if value else "No"

    try:
        if isinstance(value, (int, float)) and not pd.isna(value):
            return str(value)
    except Exception:
        pass

    text = str(value).strip()
    low = text.lower()

    yes_values = {
        "1",
        "true",
        "yes",
        "y",
        "present",
        "positive",
        "abnormal",
    }
    no_values = {
        "0",
        "false",
        "no",
        "n",
        "notpresent",
        "not present",
        "negative",
        "normal",
    }

    if low in yes_values:
        return "Yes"

    if low in no_values:
        return "No"

    if low == "good":
        return "Good"

    if low == "poor":
        return "Poor"

    if low == "ckd":
        return "CKD"

    if low in {"notckd", "not ckd", "non-ckd", "non_ckd"}:
        return "Non-CKD"

    return text


def _sentence_for_feature(feature: str, value: Any) -> str:
    """Convert one CKD feature-value pair into a natural-language sentence.

    This follows the supplied row_to_text_full() logic while preserving the
    existing one-feature-at-a-time serializer design.

    Missing values return an empty string, so they are omitted from prompts.
    """
    if _is_missing(value):
        return ""

    f = str(feature).strip()
    f_norm = _feature_key(f)
    v = _as_bool(value)

    # -------------------------
    # Literacy & lifestyle
    # -------------------------
    if f == "illiterate" or f_norm == "illiterate":
        return "The patient is illiterate." if v else "The patient is literate."

    if f == "sleeping less than 7 hours daily" or f_norm in {
        "sleeping_less_than_7_hours_daily",
        "daily_sleep_less_than_7h",
        "sleep_less_than_7h",
        "sleeping_duration_less_than_7h",
    }:
        return (
            "The patient sleeps less than seven hours per day."
            if v
            else "The patient sleeps seven or more hours per day."
        )

    if f == "tobacco smoker" or f_norm == "tobacco_smoker":
        return (
            "The patient is a tobacco smoker."
            if v
            else "The patient is not a tobacco smoker."
        )

    if f == "smokeless tobacco user" or f_norm in {
        "smokeless_tobacco_user",
        "smokeless_tobacco",
    }:
        return (
            "The patient uses smokeless tobacco."
            if v
            else "The patient does not use smokeless tobacco."
        )

    # -------------------------
    # Comorbidities
    # -------------------------
    comorbidities = {
        "has hypertension": "hypertension",
        "has_hypertension": "hypertension",
        "history_of_hypertension": "hypertension",
        "hypertension": "hypertension",
        "htn": "hypertension",
        "has diabetes": "diabetes",
        "has_diabetes": "diabetes",
        "history_of_diabetes": "diabetes",
        "diabetes": "diabetes",
        "dm": "diabetes",
        "has abdominal obesity": "abdominal obesity",
        "has_abdominal_obesity": "abdominal obesity",
        "abdominal_obesity": "abdominal obesity",
        "has undernutrition": "undernutrition",
        "has_undernutrition": "undernutrition",
        "undernutrition": "undernutrition",
        "has anemia": "anemia",
        "has_anemia": "anemia",
        "anemia": "anemia",
        "ane": "anemia",
        "has low serum albumin": "low serum albumin",
        "has_low_serum_albumin": "low serum albumin",
        "low_serum_albumin": "low serum albumin",
        "serum_albumin": "low serum albumin",
        "has hyper-cholesterolemia": "hypercholesterolemia",
        "has hypercholesterolemia": "hypercholesterolemia",
        "has_hyper_cholesterolemia": "hypercholesterolemia",
        "has_hypercholesterolemia": "hypercholesterolemia",
        "hypercholesterolemia": "hypercholesterolemia",
        "has low HDL cholesterol": "low HDL cholesterol",
        "has_low_hdl_cholesterol": "low HDL cholesterol",
        "low_hdl_cholesterol": "low HDL cholesterol",
        "hdl_cholesterol": "low HDL cholesterol",
        "has hypertriglyceridemia": "hypertriglyceridemia",
        "has_hypertriglyceridemia": "hypertriglyceridemia",
        "hypertriglyceridemia": "hypertriglyceridemia",
    }

    if f in comorbidities:
        name = comorbidities[f]
        return f"The patient has {name}." if v else f"The patient does not have {name}."

    if f_norm in comorbidities:
        name = comorbidities[f_norm]
        return f"The patient has {name}." if v else f"The patient does not have {name}."

    # -------------------------
    # Cardiovascular / renal history
    # -------------------------
    history = {
        "has history of heart disease": "a history of heart disease",
        "has_history_of_heart_disease": "a history of heart disease",
        "heart_disease": "a history of heart disease",
        "history_of_heart_disease": "a history of heart disease",
        "has history of stroke": "a history of stroke",
        "has_history_of_stroke": "a history of stroke",
        "stroke": "a history of stroke",
        "history_of_stroke": "a history of stroke",
        "has presence of red blood cells in urine": "red blood cells in the urine",
        "has_presence_of_red_blood_cells_in_urine": "red blood cells in the urine",
        "presence_of_red_blood_cells_in_urine": "red blood cells in the urine",
        "red_blood_cells_in_urine": "red blood cells in the urine",
        "rbc": "red blood cells in the urine",
    }

    if f in history:
        phrase = history[f]
        return f"The patient has {phrase}." if v else f"The patient does not have {phrase}."

    if f_norm in history:
        phrase = history[f_norm]
        return f"The patient has {phrase}." if v else f"The patient does not have {phrase}."

    # -------------------------
    # Family history
    # -------------------------
    family_history = {
        "has family history of hypertension": "hypertension",
        "has_family_history_of_hypertension": "hypertension",
        "family_history_of_hypertension": "hypertension",
        "has family history of diabetes": "diabetes",
        "has_family_history_of_diabetes": "diabetes",
        "family_history_of_diabetes": "diabetes",
        "has family history of chronic kidney disease": "chronic kidney disease",
        "has_family_history_of_chronic_kidney_disease": "chronic kidney disease",
        "family_history_of_chronic_kidney_disease": "chronic kidney disease",
        "family_history_of_ckd": "chronic kidney disease",
    }

    if f in family_history:
        disease = family_history[f]
        return (
            f"The patient has a family history of {disease}."
            if v
            else f"The patient does not have a family history of {disease}."
        )

    if f_norm in family_history:
        disease = family_history[f_norm]
        return (
            f"The patient has a family history of {disease}."
            if v
            else f"The patient does not have a family history of {disease}."
        )

    # -------------------------
    # Age group one-hot features
    # -------------------------
    age_groups = {
        "age group_18-30y": "between 18 and 30 years",
        "age_group_18_30y": "between 18 and 30 years",
        "age_group_18_30": "between 18 and 30 years",
        "age group_31-39y": "between 31 and 39 years",
        "age_group_31_39y": "between 31 and 39 years",
        "age_group_31_39": "between 31 and 39 years",
        "age group_40-49y": "between 40 and 49 years",
        "age_group_40_49y": "between 40 and 49 years",
        "age_group_40_49": "between 40 and 49 years",
        "age group_50-60y": "between 50 and 60 years",
        "age_group_50_60y": "between 50 and 60 years",
        "age_group_50_60": "between 50 and 60 years",
        "age group_60+y": "60 years or older",
        "age_group_60_plusy": "60 years or older",
        "age_group_60_plus": "60 years or older",
        "age_60y": "60 years or older",
        "age_60_plus": "60 years or older",
    }

    if f in age_groups:
        desc = age_groups[f]
        return f"The patient is {desc}." if v else f"The patient is not {desc}."

    if f_norm in age_groups:
        desc = age_groups[f_norm]
        return f"The patient is {desc}." if v else f"The patient is not {desc}."

    # -------------------------
    # Gender one-hot features
    # -------------------------
    if f in {"gender_Male", "gender_male"} or f_norm == "gender_male":
        return "The patient is male." if v else "The patient is female."

    if f in {"gender_Female", "gender_female"} or f_norm == "gender_female":
        return "The patient is female." if v else "The patient is male."

    # -------------------------
    # Marital status one-hot features
    # False values are omitted, matching row_to_text_full().
    # Supports both "maritial" typo and "marital".
    # -------------------------
    marital_status = {
        "maritial status_Married": "married",
        "marital status_Married": "married",
        "maritial_status_married": "married",
        "marital_status_married": "married",
        "maritial status_Widowed": "widowed",
        "marital status_Widowed": "widowed",
        "maritial_status_widowed": "widowed",
        "marital_status_widowed": "widowed",
        "maritial status_Others": "neither married nor widowed",
        "marital status_Others": "neither married nor widowed",
        "maritial_status_others": "neither married nor widowed",
        "marital_status_others": "neither married nor widowed",
    }

    if f in marital_status:
        return f"The patient is {marital_status[f]}." if v else ""

    if f_norm in marital_status:
        return f"The patient is {marital_status[f_norm]}." if v else ""

    # -------------------------
    # Occupation one-hot features
    # False values are omitted, matching row_to_text_full().
    # -------------------------
    occupation = {
        "occupation_Farmer": "a farmer",
        "occupation_farmer": "a farmer",
        "occupation_Housewife": "a housewife",
        "occupation_housewife": "a housewife",
        "occupation_Others": "neither a farmer nor a housewife",
        "occupation_others": "neither a farmer nor a housewife",
    }

    if f in occupation:
        return f"The patient is {occupation[f]}." if v else ""

    if f_norm in occupation:
        return f"The patient is {occupation[f_norm]}." if v else ""

    # -------------------------
    # Body mass index one-hot features
    # False values are omitted, matching row_to_text_full().
    # -------------------------
    bmi_groups = {
        "body mass index_Underweight": "underweight",
        "body_mass_index_underweight": "underweight",
        "bmi_underweight": "underweight",
        "body mass index_Normal": "of normal body mass index",
        "body_mass_index_normal": "of normal body mass index",
        "bmi_normal": "of normal body mass index",
        "body mass index_Overweight": "overweight",
        "body_mass_index_overweight": "overweight",
        "bmi_overweight": "overweight",
        "body mass index_Obese": "obese",
        "body_mass_index_obese": "obese",
        "bmi_obese": "obese",
    }

    if f in bmi_groups:
        return f"The patient is {bmi_groups[f]}." if v else ""

    if f_norm in bmi_groups:
        return f"The patient is {bmi_groups[f_norm]}." if v else ""

    # -------------------------
    # Dataset-2 / UCI CKD categorical features
    # -------------------------
    if f_norm == "pus_cell":
        return "The patient's pus cell result is abnormal." if v else "The patient's pus cell result is normal."

    if f_norm == "pus_cell_clumps":
        return "The patient has pus cell clumps." if v else "The patient does not have pus cell clumps."

    if f_norm == "bacteria":
        return "The patient has bacteria in the urine." if v else "The patient does not have bacteria in the urine."

    if f_norm == "coronary_artery_disease":
        return (
            "The patient has coronary artery disease."
            if v
            else "The patient does not have coronary artery disease."
        )

    if f_norm == "pedal_edema":
        return "The patient has pedal edema." if v else "The patient does not have pedal edema."

    # -------------------------
    # Fallback for continuous or unmatched categorical features
    # -------------------------
    display_name = DISPLAY_NAMES.get(f, f.replace("_", " ").title())
    normalized_value = normalize_value(f, value)

    if normalized_value == "":
        return ""

    return f"The patient's {display_name.lower()} is {normalized_value}."


def serialize_record_list(row: pd.Series, features: Iterable[str]) -> str:
    """Serialize a patient record as key-value pairs.

    Missing values are omitted.
    """
    parts: list[str] = []

    for feature in features:
        if feature not in row:
            continue

        value = row[feature]

        if _is_missing(value):
            continue

        display_name = DISPLAY_NAMES.get(
            str(feature),
            str(feature).replace("_", " ").title(),
        )
        normalized_value = normalize_value(str(feature), value)

        if normalized_value == "":
            continue

        parts.append(f"{display_name}: {normalized_value}")

    return ",\n".join(parts)


def serialize_record_text(row: pd.Series, features: Iterable[str]) -> str:
    """Serialize a patient record as natural-language clinical sentences.

    Missing values are omitted.
    """
    sentences: list[str] = []

    for feature in features:
        if feature not in row:
            continue

        sentence = _sentence_for_feature(str(feature), row[feature])

        if sentence:
            sentences.append(sentence)

    return " ".join(sentences)


def serialize_record(
    row: pd.Series,
    features: Iterable[str],
    style: str = "list",
) -> str:
    """Serialize a patient record using either list or text style."""
    style = style.lower().strip()

    if style in {"list", "key_value", "key-value", "kv"}:
        return serialize_record_list(row, features)

    if style in {"text", "sentence", "sentences", "narrative"}:
        return serialize_record_text(row, features)

    raise ValueError(f"Unsupported serialization style: {style}")
