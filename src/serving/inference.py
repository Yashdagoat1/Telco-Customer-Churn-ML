"""
INFERENCE PIPELINE - Production ML Model Serving (Deployment Ready)
=================================================================

Uses local artifacts:
- artifacts/model.pkl
- artifacts/feature_columns.json
"""

import os
import pandas as pd
import joblib
import json

# === PATH SETUP ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "artifacts", "feature_columns.json")

# === LOAD MODEL ===
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    raise Exception(f"❌ Failed to load model: {e}")

# === LOAD FEATURE COLUMNS ===
try:
    with open(FEATURE_PATH) as f:
        FEATURE_COLS = json.load(f)
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns")
except Exception as e:
    raise Exception(f"❌ Failed to load feature columns: {e}")

# === FEATURE TRANSFORMATION CONSTANTS ===
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # === NUMERIC CLEANING ===
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # === BINARY ENCODING ===
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .fillna(0)
                .astype(int)
            )

    # === ONE-HOT ENCODING ===
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # === BOOL → INT ===
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # === ALIGN FEATURES ===
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df


def predict(input_dict: dict) -> dict:
    """
    Predict customer churn with probability and risk level.
    """

    # Convert input to DataFrame
    df = pd.DataFrame([input_dict])

    # Transform features
    df_enc = _serve_transform(df)

    try:
        # === Prediction ===
        pred = model.predict(df_enc)[0]

        # === Probability ===
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(df_enc)[0][1])  # churn probability
        else:
            prob = None

    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    # === Risk Level Logic ===
    if prob is not None:
        if prob > 0.7:
            risk = "High"
        elif prob > 0.4:
            risk = "Medium"
        else:
            risk = "Low"
    else:
        risk = None

    # === Final Output ===
    return {
        "prediction": "Likely to churn" if pred == 1 else "Not likely to churn",
        "churn_probability": round(prob, 2) if prob is not None else None,
        "risk_level": risk
    }