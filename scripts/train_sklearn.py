# train_sklearn.py
# Train logistic regression models for diabetes and heart risk using shared utilities.

import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from app.utils import (
    DIABETES_FEATURES,
    HEART_FEATURES,
    load_data,
    prepare_xy,
    train_test_split_scaled,
    save_model_pipeline,
)

DATA_PATH = os.path.join("data", "synthetic_patients.csv")
DIABETES_MODEL_PATH = os.path.join("models", "diabetes.joblib")
HEART_MODEL_PATH = os.path.join("models", "heart.joblib")

def train_and_save(for_model="diabetes"):
    # Load dataset
    df = load_data(DATA_PATH)

    # Prepare features and labels
    X, y = prepare_xy(df, for_model=for_model)

    # Train/test split with scaling
    X_train, X_test, y_train, y_test, scaler = train_test_split_scaled(X, y)

    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"\n{for_model.capitalize()} model performance:")
    print(classification_report(y_test, y_pred))

    # Save pipeline (model + scaler + features)
    feature_names = DIABETES_FEATURES if for_model == "diabetes" else HEART_FEATURES
    model_path = DIABETES_MODEL_PATH if for_model == "diabetes" else HEART_MODEL_PATH
    save_model_pipeline(model, scaler, feature_names, model_path)
    print(f"{for_model} model trained. Accuracy={model.score(X_test, y_test):.3f}")

if __name__ == "__main__":
    train_and_save("diabetes")
    train_and_save("heart")
