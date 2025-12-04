# pages/3_ML_Model.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from utils import preprocess_for_model

st.set_page_config(page_title="ML Model · Student Productivity", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050816; color: #eaeaea; }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .section-card {
        background: #0b1120;
        padding: 1.25rem 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #1f2937;
        margin-bottom: 1.5rem;
        animation: fadeInUp 0.7s ease-out;
        animation-fill-mode: both;
    }
    .section-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.25rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Machine Learning Model")

df = st.session_state.get("df", None)
if df is None:
    st.error("No data found. Go to the **Home** page and upload the Google Forms CSV first.")
    st.stop()

# Sidebar model controls
with st.sidebar:
    st.header("Model Controls")
    model_file = st.file_uploader("Upload trained model (model.joblib)", type=["joblib"])
    train_now = st.button("Train model from current CSV")

# Persist model in session_state
if "model_data" not in st.session_state:
    st.session_state["model_data"] = None

# Load model if uploaded
if model_file is not None:
    try:
        model_data = joblib.load(model_file)
        st.session_state["model_data"] = model_data
        st.success("Model loaded successfully from uploaded file!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Train model on df
if train_now:
    with st.spinner("Training model on survey data..."):
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

        X, y, features = preprocess_for_model(df, target_col="productivity_rating")

        if pd.api.types.is_numeric_dtype(y):
            problem_type = "regression"
        else:
            problem_type = "classification"

        st.write(f"Detected problem type: **{problem_type}**")
        st.write("Features used:", features)

        if problem_type == "regression":
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            pipeline_local = Pipeline([("scaler", StandardScaler()), ("rf", model)])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            pipeline_local.fit(X_train, y_train)
            preds = pipeline_local.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            st.write(f"**RMSE:** {rmse:.3f}")
            st.write(f"**R² Score:** {r2:.3f}")
        else:
            y = y.astype(str)
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            pipeline_local = Pipeline([("scaler", StandardScaler()), ("rf", model)])
            try:
                from sklearn.model_selection import train_test_split as tts
                X_train, X_test, y_train, y_test = tts(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            pipeline_local.fit(X_train, y_train)
            preds = pipeline_local.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.write(f"**Accuracy:** {acc:.3f}")
            st.text("Classification report:")
            st.text(classification_report(y_test, preds))

        model_data = {"pipeline": pipeline_local, "features": features, "problem_type": problem_type}
        st.session_state["model_data"] = model_data
        joblib.dump(model_data, "model.joblib")
        st.success("Model trained and saved as model.joblib in the app environment.")

# Predictions + what-if
model_data = st.session_state.get("model_data", None)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Predictions & What-if Simulation</div>', unsafe_allow_html=True)

if model_data is not None:
    pipeline = model_data["pipeline"]
    features = model_data["features"]

    df_pred = df.copy()
    if "screen_time_hours" in df_pred.columns and "study_hours" in df_pred.columns:
        df_pred["screen_per_study"] = df_pred["screen_time_hours"] / (df_pred["study_hours"] + 0.1)

    missing = [f for f in features if f not in df_pred.columns]
    if missing:
        st.error(f"Missing required feature columns in data: {missing}")
    else:
        X = df_pred[features].fillna(df_pred[features].median())
        preds = pipeline.predict(X)

        st.dataframe(pd.DataFrame({"Predicted Productivity": preds}).head(20))
        st.metric("Average Predicted Productivity", f"{np.mean(preds):.2f}")

        st.subheader("🔮 What-if Simulation (Increase Study Hours)")
        pct = st.slider("Increase study hours by:", 0, 200, 20)
        X2 = X.copy()
        if "study_hours" in X2.columns:
            X2["study_hours"] *= (1 + pct / 100)
            if "screen_time_hours" in df_pred.columns and "screen_per_study" in X2.columns:
                X2["screen_per_study"] = df_pred["screen_time_hours"] / (X2["study_hours"] + 0.1)
            new_preds = pipeline.predict(X2)
            st.metric("New Predicted Avg Productivity", f"{np.mean(new_preds):.2f}")
            st.metric("Change", f"{np.mean(new_preds) - np.mean(preds):.2f}")
        else:
            st.info("No `study_hours` column available for simulation.")
else:
    st.info("Upload a model or click **Train model from current CSV** to enable predictions.")

st.markdown('</div>', unsafe_allow_html=True)
