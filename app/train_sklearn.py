# app/train_sklearn.py
# Train XGBoost ensemble models with calibration and feature selection, then save pipelines (joblib)

import os
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
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

    # ---------------- Diabetes model ----------------
    Xd, yd = prepare_xy(df, "diabetes")
    Xd_train, Xd_test, yd_train, yd_test, scaler_d = train_test_split_scaled(Xd, yd)

    # Base XGBoost
    base_xgb_d = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    # Feature selection
    selector_d = SelectFromModel(base_xgb_d, threshold="median")
    selector_d.fit(Xd_train, yd_train)
    Xd_train_sel = selector_d.transform(Xd_train)
    Xd_test_sel = selector_d.transform(Xd_test)
    selected_features_d = np.array(DIABETES_FEATURES)[selector_d.get_support()]

    # Calibrated model
    calibrated_d = CalibratedClassifierCV(base_xgb_d, cv=5)
    calibrated_d.fit(Xd_train_sel, yd_train)

    probs_d = calibrated_d.predict_proba(Xd_test_sel)[:, 1]
    auc_d = roc_auc_score(yd_test, probs_d)
    acc_d = accuracy_score(yd_test, calibrated_d.predict(Xd_test_sel))
    print(f"\nDiabetes model trained. Accuracy={acc_d:.3f}, ROC_AUC={auc_d:.3f}")
    print(classification_report(yd_test, calibrated_d.predict(Xd_test_sel)))

    save_model_pipeline(calibrated_d, scaler_d, selected_features_d.tolist(),
                        os.path.join(MODEL_DIR, "diabetes.joblib"))

    # ---------------- Heart model ----------------
    Xh, yh = prepare_xy(df, "heart")
    Xh_train, Xh_test, yh_train, yh_test, scaler_h = train_test_split_scaled(Xh, yh)

    base_xgb_h = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    selector_h = SelectFromModel(base_xgb_h, threshold="median")
    selector_h.fit(Xh_train, yh_train)
    Xh_train_sel = selector_h.transform(Xh_train)
    Xh_test_sel = selector_h.transform(Xh_test)
    selected_features_h = np.array(HEART_FEATURES)[selector_h.get_support()]

    calibrated_h = CalibratedClassifierCV(base_xgb_h, cv=5)
    calibrated_h.fit(Xh_train_sel, yh_train)

    probs_h = calibrated_h.predict_proba(Xh_test_sel)[:, 1]
    auc_h = roc_auc_score(yh_test, probs_h)
    acc_h = accuracy_score(yh_test, calibrated_h.predict(Xh_test_sel))
    print(f"\nHeart model trained. Accuracy={acc_h:.3f}, ROC_AUC={auc_h:.3f}")
    print(classification_report(yh_test, calibrated_h.predict(Xh_test_sel)))

    save_model_pipeline(calibrated_h, scaler_h, selected_features_h.tolist(),
                        os.path.join(MODEL_DIR, "heart.joblib"))

    print("\nSaved models in", MODEL_DIR)

if __name__ == "__main__":
    train_and_save_all()
