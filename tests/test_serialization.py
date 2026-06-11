from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm4ckd.features import serialize_record_list, serialize_record_text


def test_list_serialization():
    row = pd.Series({
        "history_of_hypertension": "yes",
        "history_of_diabetes": "no",
        "age": 62,
    })
    s = serialize_record_list(row, ["history_of_hypertension", "history_of_diabetes", "age"])
    assert "History of hypertension: Yes" in s
    assert "History of diabetes: No" in s
    assert "Age: 62" in s


def test_text_serialization():
    row = pd.Series({
        "history_of_hypertension": "yes",
        "history_of_diabetes": "no",
        "age": 62,
    })
    s = serialize_record_text(row, ["history_of_hypertension", "history_of_diabetes", "age"])
    assert "has history of hypertension" in s
    assert "does not have history of diabetes" in s
    assert "62 years old" in s
