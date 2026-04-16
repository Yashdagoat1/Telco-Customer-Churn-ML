"""
INFERENCE PIPELINE - Production ML Model Serving (Deployment Ready)
=================================================================

This version removes MLflow dependency and uses local artifacts:
- artifacts/model.pkl
- artifacts/feature_columns.json

Works both locally and on cloud (Render, etc.)
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


def predict(input_dict: dict) -> str:
    """
    Predict customer churn from input dictionary.
    """

    # Convert to DataFrame
    df = pd.DataFrame([input_dict])

    # Transform features
    df_enc = _serve_transform(df)

    # Predict
    try:
        preds = model.predict(df_enc)

        if hasattr(preds, "tolist"):
            preds = preds.tolist()

        result = preds[0] if isinstance(preds, list) else preds

    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    # Output
    return "Likely to churn" if result == 1 else "Not likely to churn"