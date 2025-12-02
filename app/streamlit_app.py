# app/streamlit_app.py
# Streamlit web app for interactive predictions and SHAP plots.

import sys, os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from utils import load_model_pipeline
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Streamlit page setup
st.set_page_config(page_title="Patient Risk Explorer", layout="centered")
st.title("🧠 Patient Risk Explorer — Interactive Demo")

# Prominent disclaimer under the title
st.markdown(
    """
    <div style="background-color:#ffcccc; padding:12px; border-radius:6px; text-align:center; font-size:16px; font-weight:bold;">
    ⚠️ This tool is for educational purposes only and should not be used for medical purposes.
    </div>
    """,
    unsafe_allow_html=True
)

# Custom CSS for buttons
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #1976d2;
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 1.2em;
    }
    div.stButton > button:hover {
        background-color: #1565c0;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar inputs
st.sidebar.header("Patient Inputs")
age = st.sidebar.number_input("Age", min_value=18, max_value=120, value=None, step=1, format="%d")
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=80.0, value=None, step=0.1, format="%.1f")
glucose = st.sidebar.number_input("Fasting glucose (mg/dL)", min_value=40, max_value=400, value=None, step=1, format="%d")
family_history_opt = st.sidebar.selectbox("Family history of diabetes", ["Select", "No", "Yes"])
family_history = 1 if family_history_opt == "Yes" else 0 if family_history_opt == "No" else None
activity_hours = st.sidebar.number_input("Physical activity (hrs/week)", min_value=0.0, max_value=168.0, value=None, step=0.5, format="%.1f")
systolic = st.sidebar.number_input("Systolic BP (mmHg)", min_value=60, max_value=260, value=None, step=1, format="%d")
smoking_opt = st.sidebar.selectbox("Current smoker?", ["Select", "No", "Yes"])
smoking = 1 if smoking_opt == "Yes" else 0 if smoking_opt == "No" else None
ldl = st.sidebar.number_input("LDL (mg/dL)", min_value=20, max_value=400, value=None, step=1, format="%d")

# Optional advanced inputs
st.sidebar.subheader("Optional advanced inputs")
hba1c = st.sidebar.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=None, step=0.1, format="%.1f")
waist = st.sidebar.number_input("Waist circumference (cm)", min_value=40.0, max_value=200.0, value=None, step=0.5, format="%.1f")
triglycerides = st.sidebar.number_input("Triglycerides (mg/dL)", min_value=30, max_value=600, value=None, step=1, format="%d")
medications_opt = st.sidebar.selectbox("On medication?", ["Select", "No", "Yes"])
medications = 1 if medications_opt == "Yes" else 0 if medications_opt == "No" else None

# Load trained pipelines
models_ok = True
try:
    dpipe = load_model_pipeline(str(MODEL_DIR / "diabetes.joblib"))
    hpipe = load_model_pipeline(str(MODEL_DIR / "heart.joblib"))
    model_d, scaler_d, feat_d = dpipe["model"], dpipe["scaler"], dpipe["features"]
    model_h, scaler_h, feat_h = hpipe["model"], hpipe["scaler"], hpipe["features"]
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_ok = False

# Helper: plot SHAP values as bar chart
def plot_shap_bar(shap_values, features, title):
    shap_df = pd.DataFrame({
        "Feature": features,
        "SHAP value": shap_values[0]
    }).sort_values("SHAP value", key=lambda x: abs(x), ascending=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["red" if v > 0 else "blue" for v in shap_df["SHAP value"]]
    ax.barh(shap_df["Feature"], shap_df["SHAP value"], color=colors)
    ax.set_xlabel("Impact on risk (SHAP value)")
    ax.set_title(title)
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close(fig)

# Compute risk
if models_ok and st.button("Compute risk"):
    required = [age, bmi, glucose, family_history, activity_hours, systolic, smoking, ldl]
    if None in required:
        st.warning("Please fill in all required fields before computing risk.")
    else:
        patient = {
            "age": age, "bmi": bmi, "glucose": glucose, "family_history": family_history,
            "activity_hours": activity_hours, "systolic": systolic, "smoking": smoking, "ldl": ldl,
            "hba1c": hba1c, "waist": waist, "triglycerides": triglycerides, "medications": medications
        }

        # Build feature-ordered arrays (None->0 fallback)
        Xd = np.array([[patient.get(f, 0) if patient.get(f) is not None else 0 for f in feat_d]])
        Xh = np.array([[patient.get(f, 0) if patient.get(f) is not None else 0 for f in feat_h]])

        pd_prob = float(model_d.predict_proba(scaler_d.transform(Xd))[0, 1])
        ph_prob = float(model_h.predict_proba(scaler_h.transform(Xh))[0, 1])

        st.session_state.update({
            "patient": patient,
            "pd_prob": pd_prob,
            "ph_prob": ph_prob,
            "Xd": Xd,
            "Xh": Xh,
            "feat_d": feat_d,
            "feat_h": feat_h,
            "model_d": model_d,
            "scaler_d": scaler_d,
            "model_h": model_h,
            "scaler_h": scaler_h
        })

        col1, col2 = st.columns(2)
        col1.metric("Diabetes risk", f"{pd_prob*100:.1f}%")
        col2.metric("Heart disease risk", f"{ph_prob*100:.1f}%")

        # SHAP bar chart for patient-level explanation (Diabetes)
        st.subheader("Patient-level explanation (Diabetes)")
        explainer_d = shap.TreeExplainer(model_d)
        shap_values_d = explainer_d.shap_values(scaler_d.transform(Xd))
        plot_shap_bar(shap_values_d, feat_d, "Feature contributions to diabetes risk")

# What-if scenarios
if models_ok and "patient" in st.session_state:
    st.markdown("### 🔍 What-if: Lifestyle improvements")

    with st.expander("Choose lifestyle improvements", expanded=True):
        reduce_bmi = st.checkbox("Reduce BMI by 3 units")
        lower_glucose = st.checkbox("Lower fasting glucose by 10 mg/dL")
        increase_activity = st.checkbox("Increase physical activity by 3 hrs/week")
        lower_bp = st.checkbox("Reduce systolic BP by 5 mmHg")
        quit_smoking = st.checkbox("Quit smoking")
        lower_ldl = st.checkbox("Lower LDL by 20 mg/dL")

    if st.button("Apply improvements"):
        patient = st.session_state["patient"]
        improved = patient.copy()

        if reduce_bmi and improved["bmi"] is not None:
            improved["bmi"] = max(10, improved["bmi"] - 3)
        if lower_glucose and improved["glucose"] is not None:
            improved["glucose"] = max(40, improved["glucose"] - 10)
        if increase_activity and improved["activity_hours"] is not None:
            improved["activity_hours"] = min(168, improved["activity_hours"] + 3)
        if lower_bp and improved["systolic"] is not None:
            improved["systolic"] = max(90, improved["systolic"] - 5)
        if quit_smoking:
            improved["smoking"] = 0
        if lower_ldl and improved["ldl"] is not None:
            improved["ldl"] = max(30, improved["ldl"] - 20)

        # Rebuild arrays in the same feature order
        new_Xd = np.array([[improved.get(f, 0) if improved.get(f) is not None else 0
                            for f in st.session_state["feat_d"]]])
        new_Xh = np.array([[improved.get(f, 0) if improved.get(f) is not None else 0
                            for f in st.session_state["feat_h"]]])

        new_pd = float(
            st.session_state["model_d"].predict_proba(
                st.session_state["scaler_d"].transform(new_Xd)
            )[0, 1]
        )
        new_ph = float(
            st.session_state["model_h"].predict_proba(
                st.session_state["scaler_h"].transform(new_Xh)
            )[0, 1]
        )

        st.markdown("#### Estimated risk after selected improvements:")
        col1, col2 = st.columns(2)
        col1.metric(
            "Diabetes risk",
            f"{new_pd*100:.1f}%",
            f"{st.session_state['pd_prob']*100:.1f}% → {new_pd*100:.1f}%"
        )
        col2.metric(
            "Heart disease risk",
            f"{new_ph*100:.1f}%",
            f"{st.session_state['ph_prob']*100:.1f}% → {new_ph*100:.1f}%"
        )

        # SHAP bar chart for improved patient-level explanation (Diabetes)
        st.subheader("Improved patient-level explanation (Diabetes)")
        explainer_d2 = shap.TreeExplainer(st.session_state["model_d"])
        shap_values_d2 = explainer_d2.shap_values(
            st.session_state["scaler_d"].transform(new_Xd)
        )
        plot_shap_bar(
            shap_values_d2,
            st.session_state["feat_d"],
            "Feature contributions after improvements"
        )
