import pandas as pd
from joblib import load
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
import shap, matplotlib.pyplot as plt
import os, json

# Import shared utilities
from app.utils import DIABETES_FEATURES, HEART_FEATURES, load_data, load_model_pipeline

console = Console()

# ---------------------------
# Input validation helpers
# ---------------------------
def get_int(prompt, min_val=None, max_val=None):
    while True:
        try:
            val = int(input(prompt))
            if min_val is not None and val < min_val:
                print(f"Value must be at least {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"Value must be at most {max_val}.")
                continue
            return val
        except ValueError:
            print("Please enter a valid integer.")

def get_float(prompt, min_val=None, max_val=None):
    while True:
        try:
            val = float(input(prompt))
            if min_val is not None and val < min_val:
                print(f"Value must be at least {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"Value must be at most {max_val}.")
                continue
            return val
        except ValueError:
            print("Please enter a valid number.")

# ---------------------------
# Load models and background data
# ---------------------------
diabetes_pipeline = load_model_pipeline("models/diabetes.joblib")
heart_pipeline = load_model_pipeline("models/heart.joblib")

df = load_data("data/synthetic_patients.csv")
X_d_full = df[DIABETES_FEATURES]
X_h_full = df[HEART_FEATURES]

explainer_d = shap.LinearExplainer(diabetes_pipeline["model"], X_d_full)
explainer_h = shap.LinearExplainer(heart_pipeline["model"], X_h_full)

# ---------------------------
# Explain prediction with Rich + SHAP
# ---------------------------
def explain_prediction(model_name, model, X, explainer):
    shap_values = explainer.shap_values(X)
    contribs = dict(zip(X.columns, shap_values[0]))
    total = sum(abs(v) for v in contribs.values())

    with Progress(console=console) as progress:
        task = progress.add_task(f"{model_name} Risk", total=100)
        progress.update(task, completed=100)

    table = Table(title=f"{model_name} Feature Contributions")
    table.add_column("Feature")
    table.add_column("Value", justify="right")
    table.add_column("Percent", justify="right")
    for feat, val in contribs.items():
        pct = abs(val)/total*100 if total else 0
        table.add_row(feat, f"{val:.2f}", f"{pct:.1f}%")
    console.print(table)

    shap.summary_plot(shap_values, X, show=False)
    os.makedirs("reports", exist_ok=True)
    plt.savefig(f"reports/{model_name.lower()}_shap_single.png")
    plt.close()

# ---------------------------
# Main interactive CLI
# ---------------------------
def predict_interactive():
    console.print("[bold cyan]Enter patient data[/bold cyan]")

    age = get_int("Age: ", min_val=18, max_val=120)
    bmi = get_float("BMI: ", min_val=10, max_val=60)
    glucose = get_int("Glucose: ", min_val=40, max_val=300)
    family = get_int("Family history (1=yes,0=no): ", min_val=0, max_val=1)
    activity = get_float("Weekly activity hours: ", min_val=0, max_val=80)
    systolic = get_int("Systolic BP: ", min_val=70, max_val=250)
    smoking = get_int("Smoking (1=yes,0=no): ", min_val=0, max_val=1)
    ldl = get_int("LDL cholesterol: ", min_val=30, max_val=400)

    # Diabetes
    X_d = pd.DataFrame([[age,bmi,glucose,family,activity]], columns=DIABETES_FEATURES)
    pred_d = diabetes_pipeline["model"].predict(diabetes_pipeline["scaler"].transform(X_d))[0]
    console.print(f"\nDiabetes Prediction: {'[bold red]High risk[/bold red]' if pred_d==1 else '[bold green]Low risk[/bold green]'}")
    explain_prediction("Diabetes", diabetes_pipeline["model"], X_d, explainer_d)

    # Heart
    X_h = pd.DataFrame([[age,bmi,systolic,smoking,ldl]], columns=HEART_FEATURES)
    pred_h = heart_pipeline["model"].predict(heart_pipeline["scaler"].transform(X_h))[0]
    console.print(f"\nHeart Prediction: {'[bold red]High risk[/bold red]' if pred_h==1 else '[bold green]Low risk[/bold green]'}")
    explain_prediction("Heart", heart_pipeline["model"], X_h, explainer_h)

    # Export results
    results = {
        "age": age, "bmi": bmi, "glucose": glucose, "family_history": family,
        "activity_hours": activity, "systolic": systolic, "smoking": smoking, "ldl": ldl,
        "diabetes_prediction": "High risk" if pred_d==1 else "Low risk",
        "heart_prediction": "High risk" if pred_h==1 else "Low risk"
    }
    with open("reports/patient_results.json", "w") as f:
        json.dump(results, f, indent=2)

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    predict_interactive()
