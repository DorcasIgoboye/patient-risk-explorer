# train_models.py
import os
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from app.utils import (
    ROOT, load_data, prepare_xy, save_model_pipeline,
    DIABETES_FEATURES, HEART_FEATURES
)

DATA_PATH = os.path.join(ROOT, "data", "synthetic_patients.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save(for_model, features, model_path):
    # Load data
    df = load_data(DATA_PATH)
    X, y = prepare_xy(df, for_model=for_model)

    # Split into train/test (no scaling yet)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    # Train initial model on raw features
    clf_full = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    clf_full.fit(X_train, y_train)

    # Feature selection
    selector = SelectFromModel(clf_full, threshold="median", prefit=True)
    selected_idx = selector.get_support(indices=True)
    selected_features = [features[i] for i in selected_idx]

    # Restrict train/test to selected features
    X_train_sel = X_train[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]

    # Fit scaler on selected features only
    scaler_sel = StandardScaler().fit(X_train_sel)

    # Retrain model on scaled selected features
    clf_sel = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    clf_sel.fit(scaler_sel.transform(X_train_sel), y_train)

    # Evaluate
    y_pred = clf_sel.predict_proba(scaler_sel.transform(X_test_sel))[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print(f"{for_model} AUC: {auc:.3f}")
    print(f"Selected features: {selected_features}")
    print(f"Scaler expects {scaler_sel.n_features_in_} features")

    # Save pipeline dict with model, scaler, and selected features
    save_model_pipeline(clf_sel, scaler_sel, selected_features, model_path)

if __name__ == "__main__":
    train_and_save("diabetes", DIABETES_FEATURES, os.path.join(MODEL_DIR, "diabetes.joblib"))
    train_and_save("heart", HEART_FEATURES, os.path.join(MODEL_DIR, "heart.joblib"))
