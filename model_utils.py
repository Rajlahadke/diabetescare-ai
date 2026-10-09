"""Inference utilities for the educational DiabetesCare AI demo.

Only load pickle artifacts from the project author or a trusted source.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACT_DIR = Path(__file__).resolve().parent

GENDERS = ("Female", "Male", "Other")
SMOKING_STATUSES = ("never", "No Info", "current", "former", "ever", "not current")


@dataclass(frozen=True)
class PatientProfile:
    gender: str = "Female"
    age: float = 32.0
    hypertension: bool = False
    heart_disease: bool = False
    smoking_history: str = "never"
    bmi: float = 24.8
    hba1c: float = 5.4
    glucose: float = 104.0


def load_artifacts():
    model = joblib.load(ARTIFACT_DIR / "best_diabetes_model.pkl")
    scaler = joblib.load(ARTIFACT_DIR / "scaler_diabetes.pkl")
    columns = joblib.load(ARTIFACT_DIR / "diabetes_feature_columns.pkl")

    if not isinstance(columns, (list, tuple)) or not all(isinstance(x, str) for x in columns):
        raise ValueError("Saved feature columns are not a list of strings.")
    if getattr(scaler, "n_features_in_", len(columns)) != len(columns):
        raise ValueError("Scaler feature count does not match the saved column list.")
    if getattr(model, "n_features_in_", len(columns)) != len(columns):
        raise ValueError("Model feature count does not match the saved column list.")
    expected = getattr(scaler, "feature_names_in_", None)
    if expected is not None and list(expected) != list(columns):
        raise ValueError("Scaler feature order does not match the saved column list.")
    return model, scaler, list(columns)


def encode_profile(profile: PatientProfile, columns: list[str]) -> pd.DataFrame:
    """Preserve the original training-time one-hot feature order and names."""
    if profile.gender not in GENDERS:
        raise ValueError("Unsupported gender option")
    if profile.smoking_history not in SMOKING_STATUSES:
        raise ValueError("Unsupported smoking-history option")

    raw = pd.DataFrame([{
        "gender": profile.gender,
        "age": float(profile.age),
        "hypertension": int(profile.hypertension),
        "heart_disease": int(profile.heart_disease),
        "smoking_history": profile.smoking_history,
        "bmi": float(profile.bmi),
        "HbA1c_level": float(profile.hba1c),
        "blood_glucose_level": float(profile.glucose),
    }])
    encoded = pd.get_dummies(
        raw, columns=["gender", "smoking_history"], dtype=int
    )
    # Reindex is necessary: one submitted profile can only create one dummy
    # column in each categorical group.
    return encoded.reindex(columns=columns, fill_value=0)


def predict_profile(profile: PatientProfile, model, scaler, columns: list[str]):
    encoded = encode_profile(profile, columns)
    processed = scaler.transform(encoded)
    prediction = int(np.asarray(model.predict(processed)).reshape(-1)[0])

    positive_score = None
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(processed)).reshape(1, -1)[0]
        classes = np.asarray(getattr(model, "classes_", []))
        match = np.flatnonzero(classes == 1)
        if len(match):
            positive_score = float(np.clip(probabilities[int(match[0])], 0, 1))

    return prediction, positive_score


def grouped_importances(model, columns: list[str]) -> pd.DataFrame:
    """Overall model feature importance; NOT individual patient attribution."""
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Indicator", "Importance"])
    grouping = {
        "age": "Age",
        "hypertension": "Hypertension",
        "heart_disease": "Heart disease",
        "bmi": "BMI",
        "HbA1c_level": "HbA1c",
        "blood_glucose_level": "Blood glucose",
    }
    values = {}
    for name, weight in zip(columns, model.feature_importances_):
        if name.startswith("gender_"):
            label = "Gender"
        elif name.startswith("smoking_history_"):
            label = "Smoking history"
        else:
            label = grouping.get(name, name)
        values[label] = values.get(label, 0.0) + float(weight)
    result = pd.DataFrame(
        {"Indicator": list(values), "Importance": list(values.values())}
    )
    return result.sort_values("Importance", ascending=True).reset_index(drop=True)