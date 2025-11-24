# app/train_sklearn.py
# Train logistic regression models and save pipelines (joblib)

import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from app.utils import (
    load_data,
    prepare_xy,
    train_test_split_scaled,
    save_model_pipeline,
    DIABETES_FEATURES,
    HEART_FEATURES,
)

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "data", "synthetic_patients.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save_all():
    df = load_data(DATA_PATH)

    # Diabetes
    Xd, yd = prepare_xy(df, "diabetes")
    Xd_train, Xd_test, yd_train, yd_test, scaler_d = train_test_split_scaled(Xd, yd)
    clf_d = LogisticRegression(max_iter=2000)
    clf_d.fit(Xd_train, yd_train)
    probs_d = clf_d.predict_proba(Xd_test)[:,1]
    auc_d = roc_auc_score(yd_test, probs_d)
    acc_d = accuracy_score(yd_test, clf_d.predict(Xd_test))
    print(f"\nDiabetes model trained. Accuracy={acc_d:.3f}, ROC_AUC={auc_d:.3f}")
    print(classification_report(yd_test, clf_d.predict(Xd_test)))
    save_model_pipeline(clf_d, scaler_d, DIABETES_FEATURES, os.path.join(MODEL_DIR,"diabetes.joblib"))

    # Heart
    Xh, yh = prepare_xy(df, "heart")
    Xh_train, Xh_test, yh_train, yh_test, scaler_h = train_test_split_scaled(Xh, yh)
    clf_h = LogisticRegression(max_iter=2000)
    clf_h.fit(Xh_train, yh_train)
    probs_h = clf_h.predict_proba(Xh_test)[:,1]
    auc_h = roc_auc_score(yh_test, probs_h)
    acc_h = accuracy_score(yh_test, clf_h.predict(Xh_test))
    print(f"\nHeart model trained. Accuracy={acc_h:.3f}, ROC_AUC={auc_h:.3f}")
    print(classification_report(yh_test, clf_h.predict(Xh_test)))
    save_model_pipeline(clf_h, scaler_h, HEART_FEATURES, os.path.join(MODEL_DIR,"heart.joblib"))

    print("\nSaved models in", MODEL_DIR)

if __name__ == "__main__":
    train_and_save_all()
