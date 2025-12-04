import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from streamlit_lottie import st_lottie
import requests

from utils import preprocess_for_model, extract_keywords_tfidf, basic_sentiment_score

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Student Productivity Dashboard",
    layout="wide",
    page_icon="📊"
)

# -------------------------------------------------
# LOAD LOTTIE ANIMATION
# -------------------------------------------------
def load_lottie(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None

# Super clean gradient animated analytics animation
LOTTIE_HERO = load_lottie("https://lottie.host/4d0e8b25-55d3-40b9-9689-8b5f38e1cb16/hz1gkI8j0D.json")

# -------------------------------------------------
# CSS — FULL UI UPGRADE
# -------------------------------------------------
st.markdown("""
    <style>

    /* Background + font smoothing */
    .main {
        background-color: #050816;
        color: #eaeaea;
    }
    .block-container {
        padding-top: 1.3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* Smooth fade + slide animation */
    @keyframes fadeSlideUp {
        0% { opacity: 0; transform: translateY(25px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* Hero Section Glow */
    .hero-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #111827 60%);
        padding: 2.6rem 3rem;
        border-radius: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        animation: fadeSlideUp 1s ease-out;
        box-shadow: 0px 0px 25px rgba(30, 64, 175, 0.45);
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.92;
        margin-bottom: 1rem;
    }

    /* Metric Cards — hover glow */
    .metric-card {
        background: #0f172a;
        padding: 1rem 1.2rem;
        border-radius: 0.75rem;
        border: 1px solid #1f2937;
        transition: 0.25s ease;
        animation: fadeSlideUp 0.8s ease-out;
    }
    .metric-card:hover {
        border-color: #3b82f6;
        box-shadow: 0px 0px 15px rgba(59,130,246,0.35);
        transform: translateY(-4px);
    }

    /* Section cards */
    .section-card {
        background: #0b1120;
        padding: 1.3rem 1.6rem;
        border-radius: 0.8rem;
        border: 1px solid #1f2937;
        margin-bottom: 1.5rem;
        animation: fadeSlideUp 0.7s ease;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("📊 Impact of Social Media Usage on Student Productivity")
st.write("""
This dashboard analyses **screen time**, **study hours**, **sleep** and **social media habits**
and their effect on **student productivity** using data exploration + machine learning.
""")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("Upload & Options")
    uploaded = st.file_uploader("📂 Upload Google Forms CSV", type=["csv"])
    model_file = st.file_uploader("🤖 Upload trained model (.joblib)", type=["joblib"])
    train_now = st.button("Train Model From CSV")

    st.markdown("---")
    st.caption("Built for academic research — safe, offline, ML inside the browser.")

# -------------------------------------------------
# LANDING PAGE BEFORE CSV UPLOAD
# -------------------------------------------------
if uploaded is None:
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("""
            <div class="hero-card">
                <div class="hero-title">Visualise. Analyse. Predict.</div>
                <div class="hero-subtitle">
                    Upload your Google Forms survey and unlock a complete analytical dashboard with:
                </div>
                <ul>
                    <li>📈 Interactive visual analytics</li>
                    <li>💬 Keyword + sentiment breakdown</li>
                    <li>🤖 Machine learning productivity predictor</li>
                    <li>🔮 What-if simulations</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        if LOTTIE_HERO:
            st_lottie(LOTTIE_HERO, height=320, key="hero")

    st.stop()

# -------------------------------------------------
# LOAD + CLEAN CSV
# -------------------------------------------------
df = pd.read_csv(uploaded)
df.columns = df.columns.str.strip()

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

st.success("CSV Loaded Successfully! Columns auto-mapped.")

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.subheader("Filters (EDA + Text Insights)")
df_view = df.copy()

if "gender" in df.columns:
    selected_gender = st.sidebar.multiselect(
        "Gender", df["gender"].dropna().unique(), df["gender"].dropna().unique()
    )
    df_view = df_view[df_view["gender"].isin(selected_gender)]

if "open_response" in df.columns:
    selected_purpose = st.sidebar.multiselect(
        "Purpose of Use", df["open_response"].dropna().unique(), df["open_response"].dropna().unique()
    )
    df_view = df_view[df_view["open_response"].isin(selected_purpose)]

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab_overview, tab_eda, tab_text, tab_ml = st.tabs(
    ["📋 Overview", "📊 EDA", "💬 Text Insights", "🤖 ML Model"]
)

# -------------------------------------------------
# TAB 1: OVERVIEW
# -------------------------------------------------
with tab_overview:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data Snapshot</div>', unsafe_allow_html=True)
    st.write(f"Total responses: **{len(df)}**")
    st.write(f"Filtered responses: **{len(df_view)}**")
    st.dataframe(df_view)
    st.markdown('</div>', unsafe_allow_html=True)

    # Metrics
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Summary Metrics</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Screen Time", f"{df_view['screen_time_hours'].mean():.2f} hrs")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Study Hours", f"{df_view['study_hours'].mean():.2f} hrs")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Sleep", f"{df_view['sleep_hours'].mean():.2f} hrs")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Productivity", f"{df_view['productivity_rating'].mean():.2f} / 10")
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# TAB 2: EDA
# -------------------------------------------------
with tab_eda:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Correlations</div>', unsafe_allow_html=True)

    try:
        corr = df_view[["screen_time_hours", "study_hours", "sleep_hours", "productivity_rating"]].corr()
        st.write(corr)
    except:
        st.info("Not enough numeric columns for correlation matrix.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Scatterplots
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Key Relationship Charts</div>', unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        fig1 = px.scatter(df_view, x="screen_time_hours", y="productivity_rating",
                          trendline="ols", title="Screen Time vs Productivity")
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        fig2 = px.scatter(df_view, x="study_hours", y="productivity_rating",
                          trendline="ols", title="Study Hours vs Productivity")
        st.plotly_chart(fig2, use_container_width=True)

    # Histograms + Pie Charts
    fig3 = px.histogram(df_view, x="sleep_hours", nbins=20, title="Sleep Hours Distribution")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------------------------
# TAB 3: TEXT INSIGHTS
# -------------------------------------------------
with tab_text:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Keyword & Sentiment Analysis</div>', unsafe_allow_html=True)

    if "open_response" in df_view:
        topk = st.slider("Top Keywords", 5, 30, 10)
        kw = extract_keywords_tfidf(df_view["open_response"], topk)
        st.table(pd.DataFrame(kw, columns=["Keyword", "Score"]))

        df_view["sentiment_score"] = df_view["open_response"].apply(lambda x: basic_sentiment_score(str(x)))
        st.metric("Avg Sentiment Score", f"{df_view['sentiment_score'].mean():.2f}")

        fig_sent = px.histogram(df_view, x="sentiment_score", nbins=20)
        st.plotly_chart(fig_sent, use_container_width=True)

# -------------------------------------------------
# TAB 4: MACHINE LEARNING
# -------------------------------------------------
with tab_ml:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Train or Load ML Model</div>', unsafe_allow_html=True)

    if "model_data" not in st.session_state:
        st.session_state["model_data"] = None

    # Load model
    if model_file:
        try:
            data = joblib.load(model_file)
            st.session_state["model_data"] = data
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")

    # Train new model
    if train_now:
        with st.spinner("Training ML model..."):
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_squared_error, r2_score

            X, y, features = preprocess_for_model(df)

            model = RandomForestRegressor(n_estimators=200, random_state=42)
            pipeline = Pipeline([("scaler", StandardScaler()), ("rf", model)])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, preds)):.3f}")
            st.metric("R² Score", f"{r2_score(y_test, preds):.3f}")

            st.session_state["model_data"] = {"pipeline": pipeline, "features": features}
            joblib.dump(st.session_state["model_data"], "model.joblib")

            st.success("Model trained & saved!")

    # Predictions
    if st.session_state["model_data"]:
        st.subheader("Predictions on uploaded data")
        pipeline = st.session_state["model_data"]["pipeline"]
        features = st.session_state["model_data"]["features"]

        df_pred = df.copy()
        df_pred["screen_per_study"] = df_pred["screen_time_hours"] / (df_pred["study_hours"] + 0.1)

        missing = [f for f in features if f not in df_pred.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            X = df_pred[features]
            preds = pipeline.predict(X)

            st.dataframe(pd.DataFrame({"Predicted Productivity": preds}).head(20))

            st.metric("Average Predicted Productivity", f"{np.mean(preds):.2f}")

            st.subheader("🔮 What-if Simulation")
            pct = st.slider("Increase study hours by :", 0, 200, 20)
            X2 = X.copy()
            X2["study_hours"] *= (1 + pct/100)
            new_preds = pipeline.predict(X2)

            st.metric("New Avg Product.", f"{np.mean(new_preds):.2f}")
            st.metric("Change", f"{np.mean(new_preds) - np.mean(preds):.2f}")

