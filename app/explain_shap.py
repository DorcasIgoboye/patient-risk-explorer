# app/explain_shap.py
# Compute SHAP values and save summary plots for diabetes & heart models.

import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
from app.utils import load_model_pipeline, load_data

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "data", "synthetic_patients.csv")
MODEL_DIR = os.path.join(ROOT, "models")
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

def shap_for(model_path, n=200, save_as="shap.png"):
    pipeline = load_model_pipeline(model_path)
    clf, scaler, features = pipeline["model"], pipeline["scaler"], pipeline["features"]

    df = load_data(DATA_PATH)
    X = df[features].iloc[:n]
    Xs = scaler.transform(X)

    # Use TreeExplainer for XGBoost; LinearExplainer was for logistic regression
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(Xs)

    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, save_as), dpi=150)
    plt.close()
    print("Saved", save_as)

if __name__ == "__main__":
    shap_for(os.path.join(MODEL_DIR, "diabetes.joblib"), save_as="shap_diabetes.png")
    shap_for(os.path.join(MODEL_DIR, "heart.joblib"), save_as="shap_heart.png")
