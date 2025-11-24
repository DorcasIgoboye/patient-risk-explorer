import shap, pandas as pd
from joblib import load
import matplotlib.pyplot as plt

clf = load("models/diabetes.joblib")
X = pd.read_csv("data/synthetic_patients.csv")[["age","bmi","glucose","family_history","activity_hours"]]

# New API: pass masker as second argument
masker = shap.maskers.Independent(X)
explainer = shap.LinearExplainer(clf, masker)

shap_values = explainer.shap_values(X.iloc[:50])
shap.summary_plot(shap_values, X.iloc[:50], show=False)
plt.savefig("reports/diabetes_shap_summary.png")
plt.close()
