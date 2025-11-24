import argparse
import sys
import pandas as pd
from joblib import load, dump
import shap, matplotlib.pyplot as plt
import os

# Paths
DATA = os.path.join("data", "synthetic_patients.csv")
MODELS = "models"
REPORTS = "reports"

def train():
    from train_sklearn import train_task, X_d, y_d, X_h, y_h
    train_task(X_d, y_d, "diabetes")
    train_task(X_h, y_h, "heart")

def load_model(name):
    return load(os.path.join(MODELS, f"{name}.joblib"))

def predict(name):
    model = load_model(name)
    df = pd.read_csv(DATA)
    if name == "diabetes":
        X = df[["age","bmi","glucose","family_history","activity_hours"]]
    else:
        X = df[["age","bmi","systolic","smoking","ldl"]]
    preds = model.predict(X.head(10))
    print(f"{name} predictions (first 10): {preds}")

def whatif(name):
    model = load_model(name)
    df = pd.read_csv(DATA)
    X = df.head(1)
    print(f"Baseline patient: {X.to_dict(orient='records')}")
    X_mod = X.copy()
    X_mod["bmi"] += 5
    print(f"Modified patient BMI +5 → prediction: {model.predict(X_mod)}")

def quiz():
    print("Quiz placeholder: would integrate generate_quiz here.")

def demo():
    print("Demo placeholder: run sample workflow.")

def export():
    print("Export placeholder: save outputs to reports.")

def explain(name):
    model = load_model(name)
    df = pd.read_csv(DATA)
    if name == "diabetes":
        X = df[["age","bmi","glucose","family_history","activity_hours"]]
    else:
        X = df[["age","bmi","systolic","smoking","ldl"]]
    masker = shap.maskers.Independent(X)
    explainer = shap.LinearExplainer(model, masker)
    shap_values = explainer.shap_values(X.iloc[:50])
    shap.summary_plot(shap_values, X.iloc[:50], show=False)
    plt.savefig(os.path.join(REPORTS, f"{name}_shap_summary.png"))
    plt.close()
    print(f"{name} SHAP summary saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Windows CLI for patient risk models")
    parser.add_argument("command", choices=["help","train","load","predict","whatif","quiz","demo","export","explain"])
    parser.add_argument("--model", choices=["diabetes","heart"], default="diabetes")
    args = parser.parse_args()

    if args.command == "help":
        parser.print_help()
    elif args.command == "train":
        train()
    elif args.command == "load":
        print(load_model(args.model))
    elif args.command == "predict":
        predict(args.model)
    elif args.command == "whatif":
        whatif(args.model)
    elif args.command == "quiz":
        quiz()
    elif args.command == "demo":
        demo()
    elif args.command == "export":
        export()
    elif args.command == "explain":
        explain(args.model)
