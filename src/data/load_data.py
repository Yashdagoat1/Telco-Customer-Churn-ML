import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads CSV data into a pandas DataFrame and performs basic cleaning.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)

    # 🔧 FIX: Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Handle missing values (caused by conversion)
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    return df