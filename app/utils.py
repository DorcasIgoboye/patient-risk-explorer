# app/utils.py
# Shared utilities: feature lists, preprocessing, dataset loader, pipeline save/load.

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# Root directory (project root, one level up from app/)
ROOT = os.path.dirname(os.path.dirname(__file__))

# Feature lists
DIABETES_FEATURES = ["age", "bmi", "glucose", "family_history", "activity_hours"]
HEART_FEATURES = ["age", "bmi", "systolic", "smoking", "ldl"]

def generate_synthetic(path, n=5000, seed=42):
    """Generate a synthetic patient dataset with diabetes and heart labels."""
    import random, csv, math
    random.seed(seed)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["age","bmi","glucose","family_history","activity_hours",
                  "systolic","smoking","ldl","diabetes","heart"]
        writer.writerow(header)
        for _ in range(n):
            age = int(max(18, min(90, random.gauss(50,15))))
            bmi = round(max(15.0, min(50.0, random.gauss(28,6))),1)
            glucose = int(max(60, min(200, random.gauss(100,20))))
            family = 1 if random.random() < 0.25 else 0
            activity = round(max(0.0, min(20.0, random.gauss(3,2))),1)
            systolic = int(max(90, min(200, random.gauss(130,15))))
            smoking = 1 if random.random() < 0.18 else 0
            ldl = int(max(50, min(240, random.gauss(120,30))))
            ds = 0.02*age + 0.08*bmi + 0.06*(glucose-90) + 0.9*family - 0.05*activity + random.gauss(0,0.6)
            hs = 0.025*age + 0.06*bmi + 0.03*(systolic-120) + 0.9*smoking + 0.02*(ldl-100) + random.gauss(0,0.7)
            dp = 1.0/(1.0+math.exp(-ds/10.0))
            hp = 1.0/(1.0+math.exp(-hs/10.0))
            diabetes = 1 if dp > 0.55 else 0
            heart = 1 if hp > 0.55 else 0
            writer.writerow([age,bmi,glucose,family,activity,systolic,smoking,ldl,diabetes,heart])
    print("Wrote synthetic dataset to", path)

def load_data(path):
    """Load dataset and validate required columns."""
    df = pd.read_csv(path)
    for col in ["age","bmi","glucose","family_history","activity_hours",
                "systolic","smoking","ldl","diabetes","heart"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {path}")
    return df

def prepare_xy(df, for_model="diabetes"):
    """Prepare features and labels for either diabetes or heart model."""
    if for_model == "diabetes":
        X = df[DIABETES_FEATURES].values.astype(float)
        y = df["diabetes"].values.astype(int)
    else:
        X = df[HEART_FEATURES].values.astype(float)
        y = df["heart"].values.astype(int)
    return X, y

def train_test_split_scaled(X, y, test_size=0.2, random_state=7):
    """Split into train/test sets and scale features."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler

def save_model_pipeline(model, scaler, feature_names, path):
    """Save a dict containing model, scaler, and feature names."""
    dump({"model": model, "scaler": scaler, "features": feature_names}, path)

def load_model_pipeline(path):
    """Load a saved pipeline dict. Raises error if file is not a dict."""
    obj = load(path)
    if isinstance(obj, dict) and "model" in obj and "scaler" in obj:
        return obj
    else:
        raise ValueError(f"{path} does not contain a valid model pipeline dict")

if __name__ == "__main__":
    print("Utils module loaded successfully.")
    data_path = os.path.join(ROOT, "data", "synthetic_patients.csv")
    if os.path.exists(data_path):
        df = load_data(data_path)
        print("Sample data:\n", df.head())
    else:
        print("No dataset found at", data_path)
