# app/streamlit_app.py
# Streamlit web app for interactive predictions and SHAP plots.

import streamlit as st
import numpy as np
import pandas as pd
import os
import shap
import matplotlib.pyplot as plt
from app.utils import DIABETES_FEATURES, HEART_FEATURES, load_model_pipeline

# Paths
ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# Streamlit page setup
st.set_page_config(page_title="Patient Risk Explorer", layout="centered")
st.title("Patient Risk Explorer — Interactive Demo")

# User inputs
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 120, 55)
    bmi = st.number_input("BMI", 10.0, 80.0, 28.0)
    glucose = st.number_input("Fasting glucose (mg/dL)", 40, 400, 100)
    family_history = st.selectbox("Family history of diabetes", ("No", "Yes"))
    family_history = 1 if family_history == "Yes" else 0
    activity_hours = st.number_input("Physical activity (hrs/week)", 0.0, 168.0, 3.0)
with col2:
    systolic = st.number_input("Systolic BP (mmHg)", 60, 260, 130)
    smoking = st.selectbox("Current smoker?", ("No", "Yes"))
    smoking = 1 if smoking == "Yes" else 0
    ldl = st.number_input("LDL (mg/dL)", 20, 400, 120)

# Load trained pipelines
try:
    dpipe = load_model_pipeline(os.path.join(MODEL_DIR, "diabetes.joblib"))
    hpipe = load_model_pipeline(os.path.join(MODEL_DIR, "heart.joblib"))
    model_d, scaler_d, feat_d = dpipe["model"], dpipe["scaler"], dpipe["features"]
    model_h, scaler_h, feat_h = hpipe["model"], hpipe["scaler"], hpipe["features"]
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Compute risk
if st.button("Compute risk"):
    patient = {
        "age": age, "bmi": bmi, "glucose": glucose, "family_history": family_history,
        "activity_hours": activity_hours, "systolic": systolic, "smoking": smoking, "ldl": ldl
    }
    Xd = np.array([[patient[f] for f in feat_d]])
    Xh = np.array([[patient[f] for f in feat_h]])

    pd_prob = float(model_d.predict_proba(scaler_d.transform(Xd))[0, 1])
    ph_prob = float(model_h.predict_proba(scaler_h.transform(Xh))[0, 1])
    st.metric("Diabetes risk", f"{int(pd_prob * 100)}%")
    st.metric("Heart disease risk", f"{int(ph_prob * 100)}%")

    # Contributions
    st.subheader("Feature contributions (approx.)")
    contribs_d = (model_d.coef_[0] * scaler_d.transform(Xd)[0])
    contribs_h = (model_h.coef_[0] * scaler_h.transform(Xh)[0])
    df_d = pd.DataFrame({"feature": feat_d, "value": Xd[0], "contrib": contribs_d})
    df_h = pd.DataFrame({"feature": feat_h, "value": Xh[0], "contrib": contribs_h})
    st.table(df_d)
    st.table(df_h)

    # What-if scenarios
    st.markdown("### What-if: Quick scenarios")
    if st.button("Improve lifestyle"):
        new_Xd = np.array([[age, max(10, bmi - 3), max(40, glucose - 10),
                            family_history, min(168, activity_hours + 3)]])
        new_pd = float(model_d.predict_proba(scaler_d.transform(new_Xd))[0, 1])
        st.write(f"Diabetes: {int(pd_prob * 100)}% → {int(new_pd * 100)}%")

    # SHAP summary
    if st.button("Show SHAP summary"):
        explainer_d = shap.LinearExplainer(model_d, scaler_d.transform(Xd))
        shap_values_d = explainer_d.shap_values(scaler_d.transform(Xd))
        shap.summary_plot(shap_values_d, pd.DataFrame(Xd, columns=feat_d), show=False)
        st.pyplot(plt.gcf())
        plt.close()
