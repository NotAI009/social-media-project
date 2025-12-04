import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
from utils import preprocess_for_model, extract_keywords_tfidf, basic_sentiment_score

st.set_page_config(page_title="Student Productivity Analysis Dashboard", layout="wide")

st.title("📊 Impact of Social Media Usage on Student Productivity")
st.markdown("""
This dashboard analyzes real student survey data and uses Machine Learning to explore  
how **screen time**, **study hours**, **sleep**, and **social media habits** affect **productivity**.
""")

# SIDEBAR
with st.sidebar:
    st.header("Upload & Setup")
    uploaded = st.file_uploader("Upload Google Forms CSV", type=["csv"])

    model_file = st.file_uploader("Upload trained model (model.joblib)", type=["joblib"])

    train_now = st.button("Train Model From CSV")

    target_col = "productivity_rating"

    st.markdown("---")
    st.caption("Dashboard developed for academic analysis — no external APIs used.")

# ----------------------------
# LOAD CSV
# ----------------------------
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

# ---------------------------------------------
# PREVIEW
# ---------------------------------------------
st.subheader("📄 Data Preview")
st.dataframe(df.head())

# ---------------------------------------------
# SUMMARY BOXES
# ---------------------------------------------
st.subheader("📌 Key Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg Screen Time", f"{df['screen_time_hours'].mean():.2f} hrs")
col2.metric("Avg Study Time", f"{df['study_hours'].mean():.2f} hrs")
col3.metric("Avg Sleep", f"{df['sleep_hours'].mean():.2f} hrs")
col4.metric("Avg Productivity", f"{df['productivity_rating'].mean():.2f} / 10")

# ---------------------------------------------
# EDA VISUALIZATIONS
# ---------------------------------------------
st.header("📊 Exploratory Data Analysis")

# Scatter: screen time vs productivity
fig1 = px.scatter(df, x="screen_time_hours", y="productivity_rating",
                  trendline="ols",
                  title="Screen Time vs Productivity")
st.plotly_chart(fig1, use_container_width=True)

# Scatter: study hours vs productivity
fig2 = px.scatter(df, x="study_hours", y="productivity_rating",
                  trendline="ols",
                  title="Study Hours vs Productivity")
st.plotly_chart(fig2, use_container_width=True)

# Histogram: sleep distribution
fig3 = px.histogram(df, x="sleep_hours", nbins=20,
                    title="Sleep Hours Distribution")
st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------
# NEW: Productivity vs Gender
# ----------------------------------------------------
if "gender" in df.columns:
    st.subheader("👥 Productivity by Gender")
    fig_g = px.box(df, x="gender", y="productivity_rating",
                   title="Productivity Distribution by Gender")
    st.plotly_chart(fig_g, use_container_width=True)

# ----------------------------------------------------
# NEW: Productivity vs Social Media Apps Used
# ----------------------------------------------------
if "social_apps" in df.columns:
    st.subheader("📱 Productivity by Social Media Apps Used")

    df_expanded = df.copy()
    df_expanded["app_list"] = df_expanded["social_apps"].str.split(",")

    df_exploded = df_expanded.explode("app_list")
    df_exploded["app_list"] = df_exploded["app_list"].str.strip()

    fig_apps = px.box(df_exploded,
                      x="app_list",
                      y="productivity_rating",
                      title="Productivity vs Social Media Apps")
    st.plotly_chart(fig_apps, use_container_width=True)

# ----------------------------------------------------
# NEW: Productivity vs Purpose of Usage
# ----------------------------------------------------
if "open_response" in df.columns:
    st.subheader("🎯 Productivity by Purpose of Social Media Use")

    fig_purpose = px.box(df, x="open_response", y="productivity_rating",
                         title="How Purpose of Social Media Use Affects Productivity")
    st.plotly_chart(fig_purpose, use_container_width=True)

# ----------------------------------------------------
# TEXT ANALYTICS
# ----------------------------------------------------
st.header("💬 Text Insights (Keyword Extraction + Sentiment)")

if "open_response" in df.columns:
    topk = st.slider("Select number of top keywords", 5, 30, 10)
    keywords = extract_keywords_tfidf(df["open_response"], topk)

    st.subheader("🔑 Top Keywords")
    st.table(pd.DataFrame(keywords, columns=["keyword", "score"]))

    # SENTIMENT
    df["sentiment_score"] = df["open_response"].apply(lambda x: basic_sentiment_score(str(x)))

    st.metric("Avg Sentiment Score", f"{df['sentiment_score'].mean():.2f}")

    fig_sent = px.histogram(df, x="sentiment_score", nbins=20,
                            title="Sentiment Score Distribution")
    st.plotly_chart(fig_sent, use_container_width=True)

else:
    st.info("No open text column found for insights.")

# ----------------------------------------------------
# MODEL: LOAD OR TRAIN
# ----------------------------------------------------
st.header("🤖 Machine Learning Model")

pipeline = None
features = None

if model_file is not None:
    model_data = joblib.load(model_file)
    pipeline = model_data["pipeline"]
    features = model_data["features"]
    st.success("Model loaded successfully!")

# Train model directly from CSV
if train_now:
    from train_model import train_and_save_from_df
    model_data = train_and_save_from_df(df, target_col="productivity_rating")
    pipeline = model_data["pipeline"]
    features = model_data["features"]
    st.success("Model trained & saved as model.joblib!")

# ----------------------------------------------------
# PREDICTIONS
# ----------------------------------------------------
if pipeline is not None:
    st.subheader("📈 Predictions on Uploaded Data")

    X = df[features].fillna(df[features].median())
    preds = pipeline.predict(X)

    st.dataframe(pd.DataFrame({"Predicted Productivity": preds}).head(20))

    st.metric("Average Predicted Productivity", f"{np.mean(preds):.2f}")

    # WHAT-IF SIMULATION
    st.subheader("🔮 What-if Simulation (Increase Study Hours)")

    pct = st.slider("Increase study hours by:", 0, 200, 20)
    X2 = X.copy()

    if "study_hours" in X.columns:
        X2["study_hours"] *= (1 + pct / 100)

        if "screen_time_hours" in X2.columns:
            X2["screen_per_study"] = X2["screen_time_hours"] / (X2["study_hours"] + 0.1)

        new_preds = pipeline.predict(X2)

        st.metric("New Predicted Avg Productivity", f"{np.mean(new_preds):.2f}")
        st.metric("Change", f"{np.mean(new_preds) - np.mean(preds):.2f}")

else:
    st.info("Upload a model or train one to enable predictions.")

st.markdown("---")
st.caption("Dashboard created with ❤️ for academic research.")
