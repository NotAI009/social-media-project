import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from utils import preprocess_for_model

def train_and_save_from_df(df, target_col="productivity_rating", problem_type="auto", out="model.joblib"):
    """
    Train a machine learning model (regression or classification) on uploaded CSV data.
    Automatically detects task type unless specified.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Same column mapping as in app.py
    col_map = {
        "4. Average daily social media screen time (in hours)": "screen_time_hours",
        "5. Average daily study hours (in hours)": "study_hours",
        "6. Average daily sleep hours": "sleep_hours",
        "10. Productivity Rating (1–10)": "productivity_rating",
        "8. Purpose of social media usage": "open_response",
        "7. Top social media apps you use": "social_apps",
        "3. Gender": "gender",
        "2. Age": "age",
        "1. NAME": "name",
    }
    df = df.rename(columns=col_map)

    # Preprocess
    X, y, features = preprocess_for_model(df, target_col=target_col)

    # Auto detect problem type
    if problem_type == "auto":
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 5:
            problem_type = "regression"
        else:
            problem_type = "classification"

    print("Detected problem type:", problem_type)
    print("Using features:", features)

    if problem_type == "regression":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", model)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
        print("R² Score:", r2_score(y_test, preds))

    else:  # classification
        y = y.astype(str)
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", model)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, preds))
        print(classification_report(y_test, preds))

    # Save model
    joblib.dump(
        {"pipeline": pipeline, "features": features, "problem_type": problem_type},
        out
    )
    print(f"Model saved as {out}")

    return {"pipeline": pipeline, "features": features, "problem_type": problem_type}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Impact of Social Media Usage on Student Productivity - Form Responses 1.csv")
    parser.add_argument("--target", default="productivity_rating")
    parser.add_argument("--out", default="model.joblib")
    parser.add_argument("--type", default="auto", choices=["auto", "regression", "classification"])
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    train_and_save_from_df(df, target_col=args.target, problem_type=args.type, out=args.out)
