import shap, pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import os

# Paths
DATA = os.path.join("data", "synthetic_patients.csv")
REPORTS = "reports"
os.makedirs(REPORTS, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA)

# Diabetes model
clf_diabetes = load("models/diabetes.joblib")
X_d = df[["age","bmi","glucose","family_history","activity_hours"]]

masker_d = shap.maskers.Independent(X_d)
explainer_d = shap.LinearExplainer(clf_diabetes, masker_d)
shap_values_d = explainer_d.shap_values(X_d.iloc[:50])

shap.summary_plot(shap_values_d, X_d.iloc[:50], show=False)
plt.savefig(os.path.join(REPORTS, "diabetes_shap_summary.png"))
plt.close()

print("Diabetes SHAP summary saved to reports/diabetes_shap_summary.png")

# Heart model
clf_heart = load("models/heart.joblib")
X_h = df[["age","bmi","systolic","smoking","ldl"]]

masker_h = shap.maskers.Independent(X_h)
explainer_h = shap.LinearExplainer(clf_heart, masker_h)
shap_values_h = explainer_h.shap_values(X_h.iloc[:50])

shap.summary_plot(shap_values_h, X_h.iloc[:50], show=False)
plt.savefig(os.path.join(REPORTS, "heart_shap_summary.png"))
plt.close()

print("Heart SHAP summary saved to reports/heart_shap_summary.png")
