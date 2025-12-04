import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
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
# LOTTIE HELPER (pure HTML, no extra lib)
# -------------------------------------------------
def load_lottie_json(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def lottie_html(lottie_json: dict, height: int = 260):
    if not lottie_json:
        return
    st.components.v1.html(
        f"""
        <div id="lottie-container" style="height:{height}px;"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.7.4/lottie.min.js"></script>
        <script>
        var animationData = {json.dumps(lottie_json)};
        var container = document.getElementById('lottie-container');
        if (container && animationData) {{
            lottie.loadAnimation({{
                container: container,
                renderer: 'svg',
                loop: true,
                autoplay: true,
                animationData: animationData
            }});
        }}
        </script>
        """,
        height=height + 10,
    )


# -------------------------------------------------
# SIDEBAR: THEME + UPLOADS
# -------------------------------------------------
with st.sidebar:
    st.header("Settings & Uploads")
    theme = st.selectbox("Theme", ["Dark", "Light"], index=0)

    uploaded = st.file_uploader("📂 Upload Google Forms CSV", type=["csv"])
    model_file = st.file_uploader("🤖 Upload trained model (model.joblib)", type=["joblib"])
    train_now = st.button("Train Model From CSV")

    st.markdown("---")
    st.caption("Use filters after uploading to explore different subgroups.")


# -------------------------------------------------
# THEME-DEPENDENT CSS
# -------------------------------------------------
if theme == "Dark":
    bg_main = "#050816"
    bg_card = "#0b1120"
    bg_metric = "#111827"
    text_main = "#eaeaea"
    accent = "#1d4ed8"
else:
    bg_main = "#f3f4f6"
    bg_card = "#ffffff"
    bg_metric = "#e5e7eb"
    text_main = "#111827"
    accent = "#2563eb"

st.markdown(f"""
    <style>
    .main {{
        background-color: {bg_main};
        color: {text_main};
    }}
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }}
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .section-card {{
        background: {bg_card};
        padding: 1.3rem 1.5rem;
        border-radius: 0.9rem;
        border: 1px solid #1f2937;
        margin-bottom: 1.7rem;
        animation: fadeInUp 0.7s ease-out;
    }}
    .metric-card {{
        background: {bg_metric};
        padding: 1.2rem;
        border-radius: 0.7rem;
        border: 1px solid #1f2937;
        animation: fadeInUp 0.6s ease-out;
    }}
    .metric-card:hover {{
        border-color: {accent};
        box-shadow: 0 0 15px rgba(37,99,235,0.45);
        transform: translateY(-3px);
        transition: 0.2s ease-out;
    }}
    .section-title {{
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}
    .hero-card {{
        background: radial-gradient(circle at top left, {accent}, {bg_card} 60%);
        padding: 2.4rem 3rem;
        border-radius: 1.3rem;
        border: 1px solid #1f2937;
        animation: fadeInUp 0.8s ease-out;
        margin-bottom: 1.8rem;
    }}
    .hero-title {{
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.7rem;
    }}
    .hero-subtitle {{
        opacity: 0.9;
        margin-bottom: 1rem;
    }}
    .hero-list li {{
        margin-bottom: 0.35rem;
    }}
    .chat-bubble-user {{
        background: {accent};
        color: white;
        padding: 0.6rem 0.8rem;
        border-radius: 0.8rem;
        margin-bottom: 0.4rem;
        max-width: 80%;
        align-self: flex-end;
    }}
    .chat-bubble-bot {{
        background: {bg_metric};
        color: {text_main};
        padding: 0.6rem 0.8rem;
        border-radius: 0.8rem;
        margin-bottom: 0.4rem;
        max-width: 85%;
        border: 1px solid #1f2937;
        align-self: flex-start;
    }}
    .chat-container {{
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
        margin-bottom: 0.6rem;
    }}
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HERO SECTION (TITLE + LOTTIE)
# -------------------------------------------------
st.title("📊 Impact of Social Media Usage on Student Productivity")

hero_col1, hero_col2 = st.columns([1.4, 1])

with hero_col1:
    st.markdown("""
        <div class="hero-card">
            <div class="hero-title">Visualise. Analyse. Predict.</div>
            <div class="hero-subtitle">
                This dashboard explores how social media usage, study habits and sleep
                influence students' self-reported productivity using analytics and machine learning.
            </div>
            <ul class="hero-list">
                <li>📈 Interactive charts for screen time, study hours, sleep and productivity</li>
                <li>💬 Keyword frequency & sentiment from open-ended responses</li>
                <li>🤖 Random Forest model to predict productivity scores</li>
                <li>🔮 What-if simulation: “What if students studied more?”</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with hero_col2:
    hero_lottie = load_lottie_json(
        "https://lottie.host/4d0e8b25-55d3-40b9-9689-8b5f38e1cb16/hz1gkI8j0D.json"
    )
    lottie_html(hero_lottie, height=260)

# -------------------------------------------------
# IF NO CSV → STOP
# -------------------------------------------------
if uploaded is None:
    st.info("Upload your **Google Forms CSV** in the sidebar to reveal the full dashboard.")
    st.stop()

# -------------------------------------------------
# LOAD CSV & MAP COLUMNS
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

st.success("CSV Loaded Successfully ✅ Columns mapped to internal names.")

# -------------------------------------------------
# SIDEBAR FILTERS (EDA + Text)
# -------------------------------------------------
st.sidebar.subheader("Filters (for EDA & Text tabs)")
df_view = df.copy()

if "gender" in df.columns:
    gender_filter = st.sidebar.multiselect(
        "Gender",
        options=sorted(df["gender"].dropna().unique().tolist()),
        default=sorted(df["gender"].dropna().unique().tolist()),
    )
    df_view = df_view[df_view["gender"].isin(gender_filter)]

if "open_response" in df.columns:
    purpose_filter = st.sidebar.multiselect(
        "Purpose of Social Media Use",
        options=sorted(df["open_response"].dropna().unique().tolist()),
        default=sorted(df["open_response"].dropna().unique().tolist()),
    )
    df_view = df_view[df_view["open_response"].isin(purpose_filter)]

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab_overview, tab_eda, tab_text, tab_ml = st.tabs(
    ["📋 Overview", "📊 EDA & Comparisons", "💬 Text Insights", "🤖 ML Model & Assistant"]
)

# -------------------------------------------------
# TAB 1 — OVERVIEW
# -------------------------------------------------
with tab_overview:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Summary Metrics (Filtered)</div>', unsafe_allow_html=True)

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

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data Snapshot</div>', unsafe_allow_html=True)
    st.write(f"Total responses in file: **{len(df)}**")
    st.write(f"Responses after filters: **{len(df_view)}**")
    st.dataframe(df_view)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# TAB 2 — EDA & COMPARISONS
# -------------------------------------------------
with tab_eda:
    # Small Lottie on top of EDA
    eda_lottie = load_lottie_json(
        "https://lottie.host/7bc51c50-f89c-4b24-8a40-0d0a56c2fb26/g4gD9zPvXS.json"
    )
    lottie_html(eda_lottie, height=180)

    # Correlations
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Quick Correlations</div>', unsafe_allow_html=True)
    try:
        num_cols = ["screen_time_hours", "study_hours", "sleep_hours", "productivity_rating"]
        corr = df_view[num_cols].corr()
        st.write(
            "Study hours vs productivity:",
            f"**{corr.loc['study_hours','productivity_rating']:.2f}**",
        )
        st.write(
            "Screen time vs productivity:",
            f"**{corr.loc['screen_time_hours','productivity_rating']:.2f}**",
        )
    except Exception:
        st.info("Not enough numeric data to compute correlations.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Scatter plots
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Screen Time & Study vs Productivity</div>', unsafe_allow_html=True)
    colA, colB = st.columns(2)

    with colA:
        if "screen_time_hours" in df_view.columns:
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
        if "study_hours" in df_view.columns:
            fig2 = px.scatter(
                df_view,
                x="study_hours",
                y="productivity_rating",
                trendline="ols",
                trendline_color_override="red",
                title="Study Hours vs Productivity",
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Hist + group comparisons
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sleep & Group Comparisons</div>', unsafe_allow_html=True)

    if "sleep_hours" in df_view.columns:
        fig3 = px.histogram(
            df_view,
            x="sleep_hours",
            nbins=20,
            title="Sleep Hours Distribution",
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

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# TAB 3 — TEXT INSIGHTS
# -------------------------------------------------
with tab_text:
    # small lottie
    text_lottie = load_lottie_json(
        "https://lottie.host/9f15295a-1468-4300-a6e1-5058c775cdbc/jQGGvXMEsl.json"
    )
    lottie_html(text_lottie, height=170)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Keyword & Sentiment Analysis</div>', unsafe_allow_html=True)

    if "open_response" in df_view.columns:
        topk = st.slider("Select number of top keywords", 5, 30, 10)
        keywords = extract_keywords_tfidf(df_view["open_response"], topk)
        st.subheader("🔑 Top Keywords")
        st.table(pd.DataFrame(keywords, columns=["keyword", "score"]))

        df_view["sentiment_score"] = df_view["open_response"].apply(
            lambda x: basic_sentiment_score(str(x))
        )
        st.metric("Average Sentiment Score", f"{df_view['sentiment_score'].mean():.2f}")

        fig_sent = px.histogram(
            df_view,
            x="sentiment_score",
            nbins=20,
            title="Sentiment Score Distribution",
        )
        st.plotly_chart(fig_sent, use_container_width=True)
    else:
        st.info("No `open_response` column found. Add an open text question in the survey.")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# TAB 4 — ML MODEL + GPT-STYLE ASSISTANT
# -------------------------------------------------
with tab_ml:
    # small ML lottie
    ml_lottie = load_lottie_json(
        "https://lottie.host/6a092c5d-39f6-4a68-a7c4-5f21ca9a93c1/v25G91tKk1.json"
    )
    lottie_html(ml_lottie, height=170)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Training & Predictions</div>', unsafe_allow_html=True)

    # persist model
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

            # Decide regression / classification
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

            model_data = {"pipeline": pipeline_local, "features": features, "problem_type": problem_type}
            st.session_state["model_data"] = model_data
            joblib.dump(model_data, "model.joblib")
            st.success("Model trained and saved as model.joblib in the app environment.")

    # Predictions + what-if
    model_data = st.session_state.get("model_data", None)

    if model_data is not None:
        pipeline = model_data["pipeline"]
        features = model_data["features"]

        st.subheader("📈 Predictions on Uploaded Data")

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

    # ----------------- GPT-STYLE PROFESSIONAL CHATBOT -----------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧠 Project Assistant (Professional)</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {
                "role": "assistant",
                "content": "Hello! I am your project assistant. Ask me anything about the dashboard, charts, correlations, or model."
            }
        ]

    def bot_reply(message: str) -> str:
        msg = message.lower()
        if "screen" in msg:
            return ("Higher screen time is generally associated with lower productivity "
                    "in this kind of survey. You can see this in the negative trend of the "
                    "**Screen Time vs Productivity** scatter plot.")
        if "study" in msg:
            return ("Study hours usually have a positive correlation with productivity. "
                    "In this dashboard, you can check the **Study Hours vs Productivity** chart "
                    "and the correlation value under EDA.")
        if "sleep" in msg:
            return ("Balanced sleep (around 6–8 hours) often supports better productivity. "
                    "Very low sleep can reduce focus, while extremely high sleep may indicate poor time usage.")
        if "model" in msg or "ml" in msg:
            return ("The model used here is a Random Forest, trained on numeric features like screen time, "
                    "study hours and sleep hours to predict the productivity rating. "
                    "You can retrain it using the **Train Model From CSV** button.")
        if "sentiment" in msg or "text" in msg:
            return ("The text insights tab extracts top keywords using TF-IDF and calculates a simple sentiment score "
                    "based on positive and negative terms. This helps you understand how students describe their habits.")
        if "what if" in msg or "simulation" in msg or "increase" in msg:
            return ("The what-if simulation increases study hours by a chosen percentage and recomputes predicted productivity. "
                    "It demonstrates how the model expects productivity to change if students study more.")
        if "conclusion" in msg or "summary" in msg:
            return ("A possible conclusion: higher study hours and adequate sleep tend to improve productivity, "
                    "while excessive social media screen time is associated with slightly lower self-reported productivity.")
        return ("This dashboard combines EDA, text analysis and a Random Forest model around your survey data. "
                "You can ask about screen time, study hours, sleep, the model, or how to interpret any chart, and I’ll explain it.")

    # Render chat history
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble-user">🙋‍♂️ {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    user_input = st.text_input("Ask the project assistant a question:")

    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        reply = bot_reply(user_input)
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.experimental_rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # close section-card

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("Dashboard created with ❤️ for academic research.")
