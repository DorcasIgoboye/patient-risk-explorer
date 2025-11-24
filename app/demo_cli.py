# app/demo_cli.py
# Simple demo runner that uses the saved sklearn pipelines and writes output/demo_output.txt

import os, json
import numpy as np
from app.utils import load_model_pipeline

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "demo_output.txt")

def run_demo():
    # Load full pipelines (dicts with model, scaler, features)
    diabetes_pipeline = load_model_pipeline(os.path.join(MODEL_DIR, "diabetes.joblib"))
    heart_pipeline = load_model_pipeline(os.path.join(MODEL_DIR, "heart.joblib"))

    md, sd, feat_d = diabetes_pipeline["model"], diabetes_pipeline["scaler"], diabetes_pipeline["features"]
    mh, sh, feat_h = heart_pipeline["model"], heart_pipeline["scaler"], heart_pipeline["features"]

    # Demo cases
    cases = [
        {"age":30,"bmi":22,"glucose":90,"family_history":0,"activity_hours":5,"systolic":115,"smoking":0,"ldl":100},
        {"age":55,"bmi":31,"glucose":115,"family_history":1,"activity_hours":1,"systolic":145,"smoking":1,"ldl":160},
        {"age":68,"bmi":29,"glucose":105,"family_history":0,"activity_hours":2,"systolic":150,"smoking":0,"ldl":140}
    ]

    out_lines = []
    for p in cases:
        # Build feature arrays using pipeline feature lists
        Xd = np.array([[p[f] for f in feat_d]])
        Xh = np.array([[p[f] for f in feat_h]])
        pd = md.predict_proba(sd.transform(Xd))[0,1]
        ph = mh.predict_proba(sh.transform(Xh))[0,1]
        out_lines.append("="*60)
        out_lines.append(json.dumps(p))
        out_lines.append(f"Diabetes risk: {int(pd*100)}%")
        out_lines.append(f"Heart risk: {int(ph*100)}%")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print("Wrote demo results to", OUT)

if __name__ == "__main__":
    run_demo()
