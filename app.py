import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import requests
from streamlit_lottie import st_lottie

from utils import (
    preprocess_for_model,
    extract_keywords_tfidf,
    basic_sentiment_score,
)

# ─────────────────────────────
# PAGE CONFIG
# ─────────────────────────────
st.set_page_config(
    page_title="Student Productivity Analysis Dashboard",
    layout="wide",
    page_icon="📊",
)


# ─────────────────────────────
# THEME TOGGLE (GLOBAL)
# ─────────────────────────────
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

with st.sidebar:
    st.write("### Appearance")
    use_dark = st.checkbox("Dark mode", value=st.session_state["theme"] == "dark")

st.session_state["theme"] = "dark" if use_dark else "light"

# switch plotly theme too
px.defaults.template = "plotly_dark" if st.session_state["theme"] == "dark" else "plotly_white"

# ─────────────────────────────
# CSS FOR BOTH THEMES
# ─────────────────────────────
if st.session_state["theme"] == "dark":
    bg_main = "#050816"
    text_color = "#eaeaea"
    card_bg = "#0b1120"
    metric_bg = "#020617"
else:
    bg_main = "#f3f4f6"
    text_color = "#020617"
    card_bg = "#ffffff"
    metric_bg = "#e5e7eb"

st.markdown(
    f"""
    <style>
    .main {{
        background-color: {bg_main};
        color: {text_color};
    }}
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }}
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .hero-card {{
        background: radial-gradient(circle at top left, #1d4ed8, {card_bg} 55%);
        padding: 2.5rem 3rem;
        border-radius: 1.5rem;
        border: 1px solid #1f2937;
        color: #e5e7eb;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
        animation-fill-mode: both;
    }}
    .hero-title {{
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.75rem;
    }}
    .hero-subtitle {{
        font-size: 1rem;
        opacity: 0.95;
        margin-bottom: 1.25rem;
    }}
    .hero-badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.5);
        font-size: 0.8rem;
        margin-bottom: 0.75rem;
    }}
    .hero-list li {{
        margin-bottom: 0.25rem;
    }}

    .section-card {{
        background: {card_bg};
        padding: 1.25rem 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #1f2937;
        margin-bottom: 1.5rem;
        animation: fadeInUp 0.7s ease-out;
        animation-fill-mode: both;
    }}
    .section-title {{
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }}

    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin-top: 0.75rem;
    }}
    .metric-card {{
        background: {metric_bg};
        padding: 0.9rem 1.1rem;
        border-radius: 0.75rem;
        border: 1px solid #1f2937;
        animation: fadeInUp 0.7s ease-out;
    }}
    .metric-label {{
        font-size: 0.85rem;
        opacity: 0.8;
        margin-bottom: 0.15rem;
    }}
    .metric-value {{
        font-size: 1.3rem;
        font-weight: 700;
    }}

    /* Floating chat button */
    .chat-fab {{
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 9999;
    }}
    .chat-fab button {{
        border-radius: 999px !important;
        height: 3.2rem;
        width: 3.2rem;
        padding: 0 !important;
        font-size: 1.4rem;
    }}
    .chat-panel {{
        position: fixed;
        bottom: 90px;
        right: 24px;
        width: 360px;
        max-height: 65vh;
        z-index: 9998;
        background: {card_bg};
        border-radius: 1rem;
        border: 1px solid #1f2937;
        box-shadow: 0 18px 45px rgba(0,0,0,0.45);
        padding: 0.75rem 0.9rem 0.85rem 0.9rem;
    }}
    </style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────
# LOTTIE HELPERS
# ─────────────────────────────
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


hero_anim = load_lottie_url(
    "https://assets2.lottiefiles.com/packages/lf20_touohxv0.json"
)
eda_anim = load_lottie_url(
    "https://assets8.lottiefiles.com/packages/lf20_2hu7cdvy.json"
)

# ─────────────────────────────
# TITLE
# ─────────────────────────────
st.title("📊 Impact of Social Media Usage on Student Productivity")
st.markdown(
    """
This dashboard is based on a student survey and uses data analysis plus a Machine Learning model  
to study how **screen time**, **study hours**, **sleep** and **social media habits** relate to **productivity**.
"""
)

# ─────────────────────────────
# SIDEBAR: upload + ML config
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
# LANDING SCREEN (no CSV yet)
# ─────────────────────────────
if uploaded is None:
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown(
            """
            <div class="hero-card">
                <div class="hero-badge">Step 1 · Upload the Google Forms CSV using the panel on the left</div>
                <div class="hero-title">Visualise. Analyse. Predict.</div>
                <div class="hero-subtitle">
                    This project investigates how students' social media usage is connected to their study time,
                    sleep duration and self-reported productivity.  
                    The dashboard provides a single place to explore the survey data and experiment with
                    a simple machine-learning model.
                </div>
                <ul class="hero-list">
                    <li>📈 Interactive charts for screen time, study hours, sleep and productivity</li>
                    <li>💬 Keyword frequency & basic sentiment insights from open-ended responses</li>
                    <li>🤖 A Random Forest–based model to predict productivity</li>
                    <li>🔮 What-if simulations to estimate how increased study hours may affect predicted productivity</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_right:
        if hero_anim is not None:
            st_lottie(hero_anim, height=260, key="hero")
    st.info(
        "Upload the **Google Forms CSV** from the sidebar. Once the file is loaded, the full dashboard with tabs and charts will appear."
    )
    st.caption("Dashboard created with ❤️ for academic research.")
    st.stop()

# ─────────────────────────────
# LOAD & CLEAN CSV
# ─────────────────────────────
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

st.success("CSV Loaded Successfully ✓ Columns mapped!")

# ─────────────────────────────
# SIDEBAR FILTERS (for EDA + text)
# ─────────────────────────────
st.sidebar.subheader("Filters (for EDA & text insights)")

gender_filter = None
purpose_filter = None

if "gender" in df.columns:
    gender_filter = st.sidebar.multiselect(
        "Filter by gender",
        options=sorted(df["gender"].dropna().unique().tolist()),
        default=sorted(df["gender"].dropna().unique().tolist()),
    )

if "open_response" in df.columns:
    purpose_filter = st.sidebar.multiselect(
        "Filter by purpose",
        options=sorted(df["open_response"].dropna().unique().tolist()),
        default=sorted(df["open_response"].dropna().unique().tolist()),
    )

df_view = df.copy()
if gender_filter:
    df_view = df_view[df_view["gender"].isin(gender_filter)]
if purpose_filter:
    df_view = df_view[df_view["open_response"].isin(purpose_filter)]

# ─────────────────────────────
# TABS
# ─────────────────────────────
tab_overview, tab_eda, tab_text, tab_ml = st.tabs(
    ["📋 Overview", "📊 EDA & Comparisons", "💬 Text Insights", "🤖 ML Model"]
)

# ========= OVERVIEW TAB =========
with tab_overview:
    # Hero inside overview + small lottie on right
    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown(
            """
            <div class="hero-card">
                <div class="hero-title">Visualise. Analyse. Predict.</div>
                <div class="hero-subtitle">
                    Explore how social media usage, study habits and sleep influence students' self-reported productivity.
                </div>
                <ul class="hero-list">
                    <li>📊 Clean survey snapshot</li>
                    <li>📈 Summary statistics & correlations</li>
                    <li>🤖 Model-based productivity predictions</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with colB:
        if hero_anim is not None:
            st_lottie(hero_anim, height=220, key="hero2")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Data Snapshot</div>', unsafe_allow_html=True
    )
    st.write(f"Total responses in file: **{len(df)}**")
    st.write(f"Responses after filters: **{len(df_view)}**")
    st.dataframe(df_view)
    st.markdown("</div>", unsafe_allow_html=True)

    # Clean metrics (no weird empty bars)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Summary Metrics (Filtered)</div>',
        unsafe_allow_html=True,
    )

    avg_screen = df_view["screen_time_hours"].mean()
    avg_study = df_view["study_hours"].mean()
    avg_sleep = df_view["sleep_hours"].mean()
    avg_prod = df_view["productivity_rating"].mean()

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Avg Screen Time</div>
                <div class="metric-value">{avg_screen:.2f} hrs</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Study Hours</div>
                <div class="metric-value">{avg_study:.2f} hrs</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Sleep</div>
                <div class="metric-value">{avg_sleep:.2f} hrs</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Productivity</div>
                <div class="metric-value">{avg_prod:.2f} / 10</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ========= EDA TAB =========
with tab_eda:
    top_eda = st.columns([4, 1])
    with top_eda[0]:
        st.markdown(
            '<div class="section-title">Exploratory Data Analysis</div>',
            unsafe_allow_html=True,
        )
    with top_eda[1]:
        if eda_anim is not None:
            st_lottie(eda_anim, height=120, key="eda")

    # quick correlations
    try:
        num_cols = [
            "screen_time_hours",
            "study_hours",
            "sleep_hours",
            "productivity_rating",
        ]
        corr = df_view[num_cols].corr()
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Quick Correlations</div>',
            unsafe_allow_html=True,
        )
        st.write(
            "Study hours vs productivity:",
            f"**{corr.loc['study_hours','productivity_rating']:.2f}**",
        )
        st.write(
            "Screen time vs productivity:",
            f"**{corr.loc['screen_time_hours','productivity_rating']:.2f}**",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception:
        pass

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Screen Time & Study vs Productivity</div>',
        unsafe_allow_html=True,
    )
    colA, colB = st.columns(2)
    with colA:
        fig1 = px.scatter(
            df_view,
            x="screen_time_hours",
            y="productivity_rating",
            trendline="ols",
            trendline_color_override="red",
            title="Screen Time vs Productivity",
        )
        st.plotly_chart(fig1, use_container_width=True)
    with colB:
        fig2 = px.scatter(
            df_view,
            x="study_hours",
            y="productivity_rating",
            trendline="ols",
            trendline_color_override="red",
            title="Study Hours vs Productivity",
        )
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Sleep & Group Comparisons</div>',
        unsafe_allow_html=True,
    )
    fig3 = px.histogram(
        df_view, x="sleep_hours", nbins=20, title="Sleep Hours Distribution"
    )
    st.plotly_chart(fig3, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        if "gender" in df_view.columns:
            gender_counts = df_view["gender"].value_counts()
            fig_gender = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title="Gender Distribution",
            )
            st.plotly_chart(fig_gender, use_container_width=True)
    with colD:
        if "open_response" in df_view.columns:
            purpose_counts = df_view["open_response"].value_counts()
            fig_purpose_pie = px.pie(
                values=purpose_counts.values,
                names=purpose_counts.index,
                title="Purpose of Social Media Usage",
            )
            st.plotly_chart(fig_purpose_pie, use_container_width=True)

    if "social_apps" in df_view.columns:
        df_expanded = df_view.copy()
        df_expanded["app_list"] = df_expanded["social_apps"].str.split(",")
        df_exploded = df_expanded.explode("app_list")
        df_exploded["app_list"] = df_exploded["app_list"].str.strip()
        app_counts = df_exploded["app_list"].value_counts()
        fig_apps = px.pie(
            values=app_counts.values,
            names=app_counts.index,
            title="Most Used Social Media Apps",
        )
        st.plotly_chart(fig_apps, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ========= TEXT TAB =========
with tab_text:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Keyword & Sentiment Analysis</div>',
        unsafe_allow_html=True,
    )
    if "open_response" in df_view.columns:
        topk = st.slider("Select number of top keywords", 5, 30, 10)
        keywords = extract_keywords_tfidf(df_view["open_response"], topk)
        st.subheader("🔑 Top Keywords")
        st.table(pd.DataFrame(keywords, columns=["keyword", "score"]))

        df_view["sentiment_score"] = df_view["open_response"].apply(
            lambda x: basic_sentiment_score(str(x))
        )
        st.metric("Avg Sentiment Score", f"{df_view['sentiment_score'].mean():.2f}")

        fig_sent = px.histogram(
            df_view,
            x="sentiment_score",
            nbins=20,
            title="Sentiment Score Distribution",
        )
        st.plotly_chart(fig_sent, use_container_width=True)
    else:
        st.info("No open text column found for insights.")
    st.markdown("</div>", unsafe_allow_html=True)

# ========= ML TAB =========
with tab_ml:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Model Training & Predictions</div>',
        unsafe_allow_html=True,
    )

    if "model_data" not in st.session_state:
        st.session_state["model_data"] = None

    # Load uploaded model
    if model_file is not None:
        try:
            model_data = joblib.load(model_file)
            st.session_state["model_data"] = model_data
            st.success("Model loaded successfully from uploaded file!")
        except Exception as e:
            st.error(f"Error loading model: {e}")

    # Train model from CSV
    if train_now:
        with st.spinner("Training model on uploaded CSV..."):
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import (
                mean_squared_error,
                r2_score,
                accuracy_score,
                classification_report,
            )

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
                model = RandomForestClassifier(
                    n_estimators=200, random_state=42
                )
                pipeline_local = Pipeline([("scaler", StandardScaler()), ("rf", model)])
                from sklearn.model_selection import train_test_split as tts

                try:
                    X_train, X_test, y_train, y_test = tts(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                except ValueError:
                    X_train, X_test, y_train, y_test = tts(
                        X, y, test_size=0.2, random_state=42
                    )
                pipeline_local.fit(X_train, y_train)
                preds = pipeline_local.predict(X_test)
                acc = accuracy_score(y_test, preds)
                st.write(f"**Accuracy:** {acc:.3f}")
                st.text("Classification report:")
                st.text(classification_report(y_test, preds))

            model_data = {
                "pipeline": pipeline_local,
                "features": features,
                "problem_type": problem_type,
            }
            st.session_state["model_data"] = model_data
            joblib.dump(model_data, "model.joblib")
            st.success("Model trained and saved as model.joblib in the app environment.")

    model_data = st.session_state.get("model_data", None)

    if model_data is not None:
        pipeline = model_data["pipeline"]
        features = model_data["features"]

        st.subheader("📈 Predictions on Uploaded Data")

        df_pred = df.copy()
        if "screen_time_hours" in df_pred.columns and "study_hours" in df_pred.columns:
            df_pred["screen_per_study"] = df_pred["screen_time_hours"] / (
                df_pred["study_hours"] + 0.1
            )

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
                X2["study_hours"] *= 1 + pct / 100.0
                if "screen_time_hours" in df_pred.columns and "screen_per_study" in X2.columns:
                    X2["screen_per_study"] = df_pred["screen_time_hours"] / (
                        X2["study_hours"] + 0.1
                    )
                new_preds = pipeline.predict(X2)
                st.metric(
                    "New Predicted Avg Productivity",
                    f"{np.mean(new_preds):.2f}",
                )
                st.metric(
                    "Change",
                    f"{np.mean(new_preds) - np.mean(preds):.2f}",
                )
            else:
                st.info("No `study_hours` column available for simulation.")
    else:
        st.info("Upload a model or click **Train Model From CSV** to enable predictions.")

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────
# PROJECT ASSISTANT CHATBOT (FLOATING)
# ─────────────────────────────

if "show_chat" not in st.session_state:
    st.session_state["show_chat"] = False
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I’m your project assistant. Ask me about the dashboard, charts, correlations, or model.",
        }
    ]

# Floating button
st.markdown(
    """
    <div class="chat-fab">
        <form action="#" method="post">
            <button type="submit">💬</button>
        </form>
    </div>
""",
    unsafe_allow_html=True,
)

# Streamlit can't read the HTML form, so use a normal button in an empty container
fab_col = st.empty()
if fab_col.button(" ", key="open_chat_fab"):
    st.session_state["show_chat"] = not st.session_state["show_chat"]

if st.session_state["show_chat"]:
    with st.container():
        st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
        st.markdown("#### 🤖 Project Assistant", unsafe_allow_html=True)

        for msg in st.session_state["chat_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_prompt = st.chat_input("Ask the project assistant a question")
        if user_prompt:
            st.session_state["chat_messages"].append(
                {"role": "user", "content": user_prompt}
            )

            # Simple rule-based "pro" answers
            lower = user_prompt.lower()
            if "histogram" in lower:
                answer = (
                    "A histogram groups continuous values into bins and shows how many "
                    "responses fall in each bin. In this dashboard we use it for **sleep hours** "
                    "and **sentiment scores** to see their distribution."
                )
            elif "scatter" in lower or "correlation" in lower:
                answer = (
                    "Scatter plots show the relationship between two numeric variables. "
                    "Here we plot **screen time vs productivity** and **study hours vs productivity**. "
                    "Points going upward from left to right indicate a positive correlation."
                )
            elif "model" in lower or "prediction" in lower:
                answer = (
                    "The model is a **Random Forest** using features like screen time, study hours, "
                    "sleep hours and the derived `screen_per_study`. It predicts the productivity "
                    "rating and we evaluate it using R² (for regression) or accuracy (for classification)."
                )
            elif "sentiment" in lower or "keyword" in lower:
                answer = (
                    "Text insights are based on simple TF-IDF keyword extraction and a basic "
                    "keyword sentiment score. This highlights common purposes for social media "
                    "use and whether the language is more positive or negative."
                )
            else:
                answer = (
                    "I can help explain charts, correlations, the ML model, or how to interpret results. "
                    "Try asking things like *“What does the sleep histogram show?”* or "
                    "*“How do you calculate `screen_per_study`?”*"
                )

            st.session_state["chat_messages"].append(
                {"role": "assistant", "content": answer}
            )

        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────
# FOOTER
# ─────────────────────────────
st.markdown("---")
st.caption("Dashboard created with ❤️ for academic research.")
