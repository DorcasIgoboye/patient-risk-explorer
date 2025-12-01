# train_models.py
import os
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from app.utils import (
    ROOT, load_data, prepare_xy, train_test_split_scaled, save_model_pipeline,
    DIABETES_FEATURES, HEART_FEATURES
)

DATA_PATH = os.path.join(ROOT, "data", "synthetic_patients.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save(for_model, features, model_path):
    df = load_data(DATA_PATH)
    X, y = prepare_xy(df, for_model=for_model)
    X_train_s, X_test_s, y_train, y_test, scaler = train_test_split_scaled(X, y)

    # Train XGBoost
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    clf.fit(X_train_s, y_train)

    # Feature selection
    selector = SelectFromModel(clf, threshold="median", prefit=True)
    selected_idx = selector.get_support(indices=True)
    selected_features = [features[i] for i in selected_idx]

    # Evaluate
    y_pred = clf.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print(f"{for_model} AUC: {auc:.3f}")
    print(f"Selected features: {selected_features}")

    # Save pipeline dict with model, scaler, and selected features
    save_model_pipeline(clf, scaler, selected_features, model_path)

if __name__ == "__main__":
    train_and_save("diabetes", DIABETES_FEATURES, os.path.join(MODEL_DIR, "diabetes.joblib"))
    train_and_save("heart", HEART_FEATURES, os.path.join(MODEL_DIR, "heart.joblib"))
