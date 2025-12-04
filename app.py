import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from utils import preprocess_for_model, extract_keywords_tfidf, basic_sentiment_score

st.set_page_config(page_title="Student Productivity Analysis Dashboard", layout="wide")

st.title("📊 Impact of Social Media Usage on Student Productivity")
st.markdown("""
This dashboard analyzes real student survey data and uses Machine Learning to explore  
how **screen time**, **study hours**, **sleep**, and **social media habits** affect **productivity**.
""")

# ─────────────────────────────
# SIDEBAR: upload + train button
# ─────────────────────────────
with st.sidebar:
    st.header("Upload & Setup")
    uploaded = st.file_uploader("Upload Google Forms CSV", type=["csv"])

    model_file = st.file_uploader("Upload trained model (model.joblib)", type=["joblib"])

    train_now = st.button("Train Model From CSV")

    target_col = "productivity_rating"

    st.markdown("---")
    st.caption("Dashboard developed for academic analysis — no external APIs used.")

# ─────────────────────────────
# LOAD CSV
# ─────────────────────────────
df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)

    # Strip extra spaces in column names
    df.columns = df.columns.str.strip()

    # MAP YOUR REAL CSV COLUMNS → CLEAN NAMES
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

    st.success("CSV Loaded Successfully ✓ Columns mapped!")

if df is None:
    st.info("Please upload your real CSV file to continue.")
    st.stop()

# ─────────────────────────────
# SIDEBAR FILTERS (for EDA only)
# ─────────────────────────────
st.sidebar.subheader("Filters (for EDA & text insights)")

gender_filter = None
purpose_filter = None

if "gender" in df.columns:
    gender_filter = st.sidebar.multiselect(
        "Filter by gender",
        options=sorted(df["gender"].dropna().unique().tolist()),
        default=sorted(df["gender"].dropna().unique().tolist())
    )

if "open_response" in df.columns:
    purpose_filter = st.sidebar.multiselect(
        "Filter by purpose",
        options=sorted(df["open_response"].dropna().unique().tolist()),
        default=sorted(df["open_response"].dropna().unique().tolist())
    )

# apply filters to create df_view (used for EDA)
df_view = df.copy()
if gender_filter:
    df_view = df_view[df_view["gender"].isin(gender_filter)]
if purpose_filter:
    df_view = df_view[df_view["open_response"].isin(purpose_filter)]

# ─────────────────────────────
# PREVIEW
# ─────────────────────────────
st.subheader("📄 Data Preview")
st.write(f"Total responses in file: {len(df)}")
st.write(f"Responses after filters: {len(df_view)}")
st.dataframe(df_view)

# ─────────────────────────────
# SUMMARY BOXES (based on filtered data)
# ─────────────────────────────
st.subheader("📌 Key Summary Statistics (Filtered)")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg Screen Time", f"{df_view['screen_time_hours'].mean():.2f} hrs")
col2.metric("Avg Study Time", f"{df_view['study_hours'].mean():.2f} hrs")
col3.metric("Avg Sleep", f"{df_view['sleep_hours'].mean():.2f} hrs")
col4.metric("Avg Productivity", f"{df_view['productivity_rating'].mean():.2f} / 10")

# ─────────────────────────────
# EDA VISUALIZATIONS (use df_view)
# ─────────────────────────────
st.header("📊 Exploratory Data Analysis")

# Scatter: screen time vs productivity
fig1 = px.scatter(
    df_view,
    x="screen_time_hours",
    y="productivity_rating",
    title="Screen Time vs Productivity"
)
st.plotly_chart(fig1, use_container_width=True)

# Scatter: study hours vs productivity
fig2 = px.scatter(
    df_view,
    x="study_hours",
    y="productivity_rating",
    title="Study Hours vs Productivity"
)
st.plotly_chart(fig2, use_container_width=True)

# Histogram: sleep distribution
fig3 = px.histogram(
    df_view,
    x="sleep_hours",
    nbins=20,
    title="Sleep Hours Distribution"
)
st.plotly_chart(fig3, use_container_width=True)

# Productivity vs Gender
if "gender" in df_view.columns:
    st.subheader("👥 Productivity by Gender")
    fig_g = px.box(
        df_view,
        x="gender",
        y="productivity_rating",
        title="Productivity Distribution by Gender"
    )
    st.plotly_chart(fig_g, use_container_width=True)

# Productivity vs Social Media Apps Used
if "social_apps" in df_view.columns:
    st.subheader("📱 Productivity by Social Media Apps Used")

    df_expanded = df_view.copy()
    df_expanded["app_list"] = df_expanded["social_apps"].str.split(",")

    df_exploded = df_expanded.explode("app_list")
    df_exploded["app_list"] = df_exploded["app_list"].str.strip()

    fig_apps = px.box(
        df_exploded,
        x="app_list",
        y="productivity_rating",
        title="Productivity vs Social Media Apps"
    )
    st.plotly_chart(fig_apps, use_container_width=True)

# Productivity vs Purpose of Usage
if "open_response" in df_view.columns:
    st.subheader("🎯 Productivity by Purpose of Social Media Use")

    fig_purpose = px.box(
        df_view,
        x="open_response",
        y="productivity_rating",
        title="How Purpose of Social Media Use Affects Productivity"
    )
    st.plotly_chart(fig_purpose, use_container_width=True)

# ─────────────────────────────
# TEXT ANALYTICS (use df_view)
# ─────────────────────────────
st.header("💬 Text Insights (Keyword Extraction + Sentiment)")

if "open_response" in df_view.columns:
    topk = st.slider("Select number of top keywords", 5, 30, 10)
    keywords = extract_keywords_tfidf(df_view["open_response"], topk)

    st.subheader("🔑 Top Keywords")
    st.table(pd.DataFrame(keywords, columns=["keyword", "score"]))

    # SENTIMENT
    df_view["sentiment_score"] = df_view["open_response"].apply(lambda x: basic_sentiment_score(str(x)))

    st.metric("Avg Sentiment Score", f"{df_view['sentiment_score'].mean():.2f}")

    fig_sent = px.histogram(
        df_view,
        x="sentiment_score",
        nbins=20,
        title="Sentiment Score Distribution"
    )
    st.plotly_chart(fig_sent, use_container_width=True)

else:
    st.info("No open text column found for insights.")

# ─────────────────────────────
# MODEL: LOAD OR TRAIN (uses full df, not filtered)
# ─────────────────────────────
# ----------------------------------------------------
# MODEL: LOAD OR TRAIN (uses full df, not filtered)
# ----------------------------------------------------
st.header("🤖 Machine Learning Model")

# 🔹 Persist model across reruns
if "model_data" not in st.session_state:
    st.session_state["model_data"] = None

# 1) If user uploads an already trained model.joblib
if model_file is not None:
    try:
        model_data = joblib.load(model_file)
        st.session_state["model_data"] = model_data
        st.success("Model loaded successfully from uploaded file!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# 2) Train model directly from the currently loaded CSV
if train_now:
    with st.spinner("Training model on uploaded CSV..."):
        # Import here so app still loads even if sklearn missing at import time
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

        # Use same pre-processing as everywhere (on full df)
        X, y, features = preprocess_for_model(df, target_col="productivity_rating")

        # For this project: productivity_rating is numeric (1–10) so treat as regression
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

        else:  # (kept for future classification use)
            y = y.astype(str)
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            pipeline_local = Pipeline([("scaler", StandardScaler()), ("rf", model)])

            try:
                X_train, X_test, y_train, y_test = train_test_split(
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

        # Save model data in session_state so it survives reruns
        model_data = {
            "pipeline": pipeline_local,
            "features": features,
            "problem_type": problem_type,
        }
        st.session_state["model_data"] = model_data

        # (Optional) also save to disk
        joblib.dump(model_data, "model.joblib")
        st.success("Model trained and saved as model.joblib in the app environment.")

# ----------------------------------------------------
# PREDICTIONS (use model from session_state)
# ----------------------------------------------------
model_data = st.session_state.get("model_data", None)

if model_data is not None:
    pipeline = model_data["pipeline"]
    features = model_data["features"]

    st.subheader("📈 Predictions on Uploaded Data")

    # Recreate derived feature so 'screen_per_study' exists if model expects it
    df_pred = df.copy()
    if "screen_time_hours" in df_pred.columns and "study_hours" in df_pred.columns:
        df_pred["screen_per_study"] = df_pred["screen_time_hours"] / (df_pred["study_hours"] + 0.1)

    # Make sure required features exist
    missing = [f for f in features if f not in df_pred.columns]
    if missing:
        st.error(f"Missing required feature columns in data: {missing}")
    else:
        X = df_pred[features].fillna(df_pred[features].median())
        preds = pipeline.predict(X)

        st.dataframe(pd.DataFrame({"Predicted Productivity": preds}).head(20))
        st.metric("Average Predicted Productivity", f"{np.mean(preds):.2f}")

        # WHAT-IF SIMULATION
        st.subheader("🔮 What-if Simulation (Increase Study Hours)")
        pct = st.slider("Increase study hours by:", 0, 200, 20)
        X2 = X.copy()

        if "study_hours" in X2.columns:
            X2["study_hours"] *= (1 + pct / 100)

            if "screen_time_hours" in df_pred.columns and "screen_per_study" in X2.columns:
                # recompute screen_per_study using original screen_time_hours and new study_hours
                X2["screen_per_study"] = df_pred["screen_time_hours"] / (X2["study_hours"] + 0.1)

            new_preds = pipeline.predict(X2)

            st.metric("New Predicted Avg Productivity", f"{np.mean(new_preds):.2f}")
            st.metric("Change", f"{np.mean(new_preds) - np.mean(preds):.2f}")
        else:
            st.info("No `study_hours` column available for simulation.")
else:
    st.info("Upload a model or click **Train Model From CSV** to enable predictions.")

st.markdown("---")
st.caption("Dashboard created with ❤️ for academic research.")
