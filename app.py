import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import requests
import json

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
# CSS + ANIMATIONS
# -------------------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #050816;
        color: #eaeaea;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .section-card {
        background: #0b1120;
        padding: 1.3rem 1.5rem;
        border-radius: 0.9rem;
        border: 1px solid #1f2937;
        margin-bottom: 1.7rem;
        animation: fadeInUp 0.7s ease-out;
    }
    .metric-card {
        background: #111827;
        padding: 1.2rem;
        border-radius: 0.7rem;
        border: 1px solid #1f2937;
        animation: fadeInUp 0.6s ease-out;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .hero-card {
        background: radial-gradient(circle at top left, #1d4ed8, #0b1120 60%);
        padding: 2.4rem 3rem;
        border-radius: 1.3rem;
        border: 1px solid #1f2937;
        animation: fadeInUp 0.8s ease-out;
        margin-bottom: 1.8rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.7rem;
    }
    .hero-subtitle {
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    .hero-list li {
        margin-bottom: 0.35rem;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOTTIE HELPER
# -------------------------------------------------
def load_lottie(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None


# Working animation link
LOTTIE_URL = "https://lottie.host/3e38f48b-1c80-4b45-b879-4bb2db15d2af/7tSYQ7YxOj.json"
lottie_anim = load_lottie(LOTTIE_URL)

# -------------------------------------------------
# HERO SECTION
# -------------------------------------------------
col1, col2 = st.columns([1.4, 1])

with col1:
    st.markdown("""
        <div class="hero-card">
            <div class="hero-title">Visualise. Analyse. Predict.</div>
            <div class="hero-subtitle">
                Discover how social media habits shape study time, sleep, and productivity.
                This dashboard combines analytics + ML simulations for maximum insight.
            </div>
            <ul class="hero-list">
                <li>📊 Deep-dive analytics & comparisons</li>
                <li>💬 Keyword & sentiment insights</li>
                <li>🤖 ML predictions for productivity</li>
                <li>🔮 What-if simulations for study improvement</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with col2:
    if lottie_anim:
        from streamlit_lottie import st_lottie
        st_lottie(lottie_anim, height=280, key="hero_anim")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("Upload Data")
    uploaded = st.file_uploader("Upload Google Forms CSV", type=["csv"])

    model_file = st.file_uploader("Upload Model (model.joblib)", type=["joblib"])
    train_now = st.button("Train Model From CSV")
    st.markdown("---")
    st.caption("Dashboard created for academic analysis only.")

# -------------------------------------------------
# STOP IF NO CSV
# -------------------------------------------------
if uploaded is None:
    st.info("Upload your **Google Forms CSV** in the sidebar to begin.")
    st.stop()

# -------------------------------------------------
# LOAD CSV + CLEANING
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

st.success("CSV Loaded Successfully ✓ Columns mapped")

# -------------------------------------------------
# FILTERS
# -------------------------------------------------
st.sidebar.subheader("Filters")

df_view = df.copy()

if "gender" in df.columns:
    gf = st.sidebar.multiselect(
        "Filter by Gender",
        options=df["gender"].dropna().unique().tolist(),
        default=df["gender"].dropna().unique().tolist()
    )
    df_view = df_view[df_view["gender"].isin(gf)]

if "open_response" in df.columns:
    pf = st.sidebar.multiselect(
        "Filter by Purpose",
        options=df["open_response"].dropna().unique().tolist(),
        default=df["open_response"].dropna().unique().tolist()
    )
    df_view = df_view[df_view["open_response"].isin(pf)]

# -------------------------------------------------
# TABS  (ADDED 2 MATH TABS)
# -------------------------------------------------
tab_overview, tab_eda, tab_text, tab_ml, tab_math1, tab_math2 = st.tabs(
    ["📋 Overview", "📊 EDA", "💬 Text Insights", "🤖 ML Model", "📐 Matrix & Vectors", "∫ Integration & Math Insights"]
)

# -------------------------------------------------
# TAB 1 — OVERVIEW
# -------------------------------------------------
with tab_overview:
    st.markdown('<div class="section-card"><div class="section-title">Summary Metrics</div>', unsafe_allow_html=True)

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

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">Data Snapshot</div>', unsafe_allow_html=True)
    st.dataframe(df_view)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# TAB 2 — EDA
# -------------------------------------------------
with tab_eda:
    st.markdown('<div class="section-card"><div class="section-title">Correlations</div>', unsafe_allow_html=True)
    try:
        corr = df_view[["screen_time_hours", "study_hours", "sleep_hours", "productivity_rating"]].corr()
        st.write("Study Hours ↗ Productivity:", f"**{corr.loc['study_hours', 'productivity_rating']:.2f}**")
        st.write("Screen Time ↘ Productivity:", f"**{corr.loc['screen_time_hours', 'productivity_rating']:.2f}**")
    except:
        pass
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">Charts</div>', unsafe_allow_html=True)
    colA, colB = st.columns(2)

    with colA:
        fig1 = px.scatter(df_view, x="screen_time_hours", y="productivity_rating",
                          trendline="ols", title="Screen Time vs Productivity")
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        fig2 = px.scatter(df_view, x="study_hours", y="productivity_rating",
                          trendline="ols", title="Study Hours vs Productivity")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# TAB 3 — TEXT INSIGHTS
# -------------------------------------------------
with tab_text:
    st.markdown('<div class="section-card"><div class="section-title">Keyword & Sentiment Analysis</div>', unsafe_allow_html=True)

    if "open_response" in df_view.columns:
        topk = st.slider("Top Keywords", 5, 30, 10)
        words = extract_keywords_tfidf(df_view["open_response"], topk)
        st.table(pd.DataFrame(words, columns=["Keyword", "Score"]))

        df_view["sentiment"] = df_view["open_response"].apply(lambda x: basic_sentiment_score(str(x)))
        st.metric("Average Sentiment Score", f"{df_view['sentiment'].mean():.2f}")

        fig_sent = px.histogram(df_view, x="sentiment", nbins=20, title="Sentiment Distribution")
        st.plotly_chart(fig_sent, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# TAB 4 — ML MODEL
# -------------------------------------------------
with tab_ml:
    st.markdown('<div class="section-card"><div class="section-title">Model Training & Predictions</div>', unsafe_allow_html=True)

    if "model_data" not in st.session_state:
        st.session_state["model_data"] = None

    if model_file:
        try:
            model_data = joblib.load(model_file)
            st.session_state["model_data"] = model_data
            st.success("Model Loaded Successfully")
        except Exception as e:
            st.error(f"Error loading: {e}")

    if train_now:
        with st.spinner("Training model..."):
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_squared_error, r2_score

            X, y, features = preprocess_for_model(df)

            model = RandomForestRegressor(n_estimators=200, random_state=42)
            pipe = Pipeline([("scale", StandardScaler()), ("rf", model)])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            st.write("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
            st.write("R² Score:", r2_score(y_test, preds))

            st.session_state["model_data"] = {"pipeline": pipe, "features": features}
            joblib.dump(st.session_state["model_data"], "model.joblib")

            st.success("Model Trained & Saved")

    model_data = st.session_state["model_data"]

    if model_data:
        pipe = model_data["pipeline"]
        features = model_data["features"]

        df_temp = df.copy()
        if "screen_time_hours" in df_temp and "study_hours" in df_temp:
            df_temp["screen_per_study"] = df_temp["screen_time_hours"] / (df_temp["study_hours"] + 0.1)

        X = df_temp[features].fillna(df_temp[features].median())
        predictions = pipe.predict(X)

        st.subheader("Predicted Productivity")
        st.dataframe(pd.DataFrame({"Predicted Productivity": predictions}))

        st.metric("Average Prediction", f"{np.mean(predictions):.2f}")

# -------------------------------------------------
# TAB 5 — MATRIX & VECTORS
# -------------------------------------------------
with tab_math1:
    st.markdown('<div class="section-card"><div class="section-title">📐 Matrix & Vector Representation</div>', unsafe_allow_html=True)
    st.write("""
    This section shows how the survey data is represented using a **feature matrix** and
    **feature vectors**, matching the **Matrices** and **Vector Algebra** chapters.
    """)

    try:
        # Use same preprocessing as ML model
        X_math, y_math, feature_cols_math = preprocess_for_model(df_view, target_col="productivity_rating")

        st.subheader("1️⃣ Feature Matrix (X)")
        st.write("Each row = one student, each column = one numeric feature used by the model.")
        st.latex(r"X = \begin{bmatrix} x_{11} & x_{12} & \dots & x_{1n} \\ x_{21} & x_{22} & \dots & x_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ x_{m1} & x_{m2} & \dots & x_{mn} \end{bmatrix}")
        st.dataframe(X_math)

        st.subheader("2️⃣ Feature Vector for Selected Student")
        if len(X_math) > 0:
            idx = st.number_input("Select student index (row in matrix):", 0, len(X_math) - 1, 0, 1)
            vec = X_math.iloc[int(idx)].values
            st.write("Feature vector for that student:")
            st.latex(r"\vec{v} = \left[" + " \; ".join([str(round(v, 2)) for v in vec]) + r"\right]")
            st.write("This is the vector the ML model uses as input for prediction.")
        else:
            st.info("Not enough data to display matrix and vectors.")
    except Exception as e:
        st.warning(f"Could not create matrix/vector view: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# TAB 6 — INTEGRATION & MATH INSIGHTS
# -------------------------------------------------
with tab_math2:
    st.markdown('<div class="section-card"><div class="section-title">∫ Integration & Productivity Curve</div>', unsafe_allow_html=True)
    st.write("""
    Here we use **numerical integration (trapezoidal rule)** to approximate the area under the  
    **Productivity vs Study Hours** curve, which creates a single **Productivity Index**.
    """)

    if "study_hours" in df_view.columns and "productivity_rating" in df_view.columns:
        df_int = df_view.dropna(subset=["study_hours", "productivity_rating"]).copy()
        if len(df_int) >= 2:
            df_int = df_int.sort_values("study_hours")
            x = df_int["study_hours"].values
            y = df_int["productivity_rating"].values

            # Trapezoidal numerical integration
            area = np.trapz(y, x)

            st.latex(r"\text{Productivity Index} \approx \int f(t)\,dt \approx \sum \frac{(x_{i+1}-x_i)\,[f(x_i)+f(x_{i+1})]}{2}")
            st.metric("Integration-based Productivity Index", f"{area:.2f}")

            fig_int = px.area(df_int, x="study_hours", y="productivity_rating",
                              title="Area under Productivity vs Study Hours (Numerical Integration)")
            st.plotly_chart(fig_int, use_container_width=True)
        else:
            st.info("Not enough distinct points to perform integration meaningfully.")
    else:
        st.info("Need 'study_hours' and 'productivity_rating' columns to compute integral.")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("Dashboard created with ❤️ for academic research.")
