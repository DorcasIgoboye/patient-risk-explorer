# app/explain_shap.py
# Compute SHAP values and save summary plots for diabetes & heart models.

import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
from app.utils import DIABETES_FEATURES, HEART_FEATURES, load_model_pipeline

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "data", "synthetic_patients.csv")
MODEL_DIR = os.path.join(ROOT, "models")
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

def shap_for(model_path, feature_names, n=200, save_as="shap.png"):
    pipeline = load_model_pipeline(model_path)
    clf, scaler = pipeline["model"], pipeline["scaler"]

    df = pd.read_csv(DATA_PATH)
    X = df[feature_names].iloc[:n]
    Xs = scaler.transform(X)

    # Linear explainer for logistic regression
    explainer = shap.LinearExplainer(clf, Xs)
    shap_values = explainer.shap_values(Xs)

    # Summary plot
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, save_as), dpi=150)
    plt.close()
    print("Saved", save_as)

if __name__ == "__main__":
    shap_for(os.path.join(MODEL_DIR, "diabetes.joblib"), DIABETES_FEATURES, save_as="shap_diabetes.png")
    shap_for(os.path.join(MODEL_DIR, "heart.joblib"), HEART_FEATURES, save_as="shap_heart.png")
