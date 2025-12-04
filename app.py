import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import json
import requests

from utils import extract_keywords_tfidf, basic_sentiment_score, preprocess_for_model


# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Student Productivity Dashboard",
                   layout="wide",
                   page_icon="📊")


# -----------------------
# LOTTIE LOADER (SAFE)
# -----------------------
def load_lottie(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return {}
    return {}


lottie_data = load_lottie("https://assets2.lottiefiles.com/packages/lf20_totrpclr.json")


# -----------------------
# CSS (NEON + CLEAN)
# -----------------------
st.markdown("""
<style>
.main { background-color: #050816; color: #eaeaea; }
.block-container { padding-top: 1rem; }
.card {
    background: #0b1120;
    padding: 1.4rem;
    border-radius: 0.7rem;
    border: 1px solid #1f2937;
    margin-bottom: 1.7rem;
}
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# -----------------------
# HERO HEADER
# -----------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.title("📱 Impact of Social Media on Student Productivity")
    st.write("""
Upload your Google Forms CSV to get:

- 📊 Interactive screen time, study, sleep analytics  
- 💬 Keyword & sentiment insights  
- 🤖 ML predictions for productivity  
- 🎯 What-if study simulations  
""")

with col2:
    st.components.v1.html(
        f"""
        <script src="https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.7.4/lottie.min.js"></script>
        <div id="lottie" style="height:240px;"></div>
        <script>
        lottie.loadAnimation({{
            container: document.getElementById('lottie'),
            renderer: 'svg',
            loop: true,
            autoplay: true,
            animationData: {json.dumps(lottie_data)}
        }});
        </script>
        """,
        height=260,
    )

st.divider()


# -----------------------
# UPLOAD CSV
# -----------------------
uploaded = st.file_uploader("📂 Upload Your Google Forms CSV", type=["csv"])

if uploaded is None:
    st.info("⬆️ Upload your CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)
df.columns = df.columns.str.strip()

# Map Google Form column names → clean names used by ML
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

st.success("CSV loaded and mapped successfully! 🎉")

st.divider()


# -----------------------
# SECTION 1 — DATA SNAPSHOT
# -----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Data Snapshot</div>', unsafe_allow_html=True)

st.write(f"Total responses: **{len(df)}**")
st.dataframe(df)
st.markdown('</div>', unsafe_allow_html=True)


# -----------------------
# SECTION 2 — DASHBOARD VISUALS
# -----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📈 Dashboard Visuals</div>', unsafe_allow_html=True)

if "screen_time_hours" in df.columns and "productivity_rating" in df.columns:
    fig1 = px.scatter(df, x="screen_time_hours", y="productivity_rating",
                      trendline="ols", title="Screen Time vs Productivity")
    st.plotly_chart(fig1, use_container_width=True)

if "study_hours" in df.columns and "productivity_rating" in df.columns:
    fig2 = px.scatter(df, x="study_hours", y="productivity_rating",
                      trendline="ols", title="Study Hours vs Productivity")
    st.plotly_chart(fig2, use_container_width=True)

if "sleep_hours" in df.columns:
    fig3 = px.histogram(df, x="sleep_hours", nbins=20, title="Sleep Hours Distribution")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# -----------------------
# SECTION 3 — TEXT INSIGHTS
# -----------------------
if "open_response" in df.columns:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💬 Text Insights</div>', unsafe_allow_html=True)

    keywords = extract_keywords_tfidf(df["open_response"], topk=10)
    st.subheader("🔑 Top Keywords")
    st.table(pd.DataFrame(keywords, columns=["Keyword", "Score"]))

    df["sentiment_score"] = df["open_response"].apply(lambda x: basic_sentiment_score(str(x)))
    st.metric("Average Sentiment Score", f"{df['sentiment_score'].mean():.2f}")

    fig_sent = px.histogram(df, x="sentiment_score", nbins=20,
                            title="Sentiment Score Distribution")
    st.plotly_chart(fig_sent, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------
# SECTION 4 — MACHINE LEARNING
# -----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🤖 ML Model & Predictions</div>', unsafe_allow_html=True)

train_btn = st.button("Train ML Model")

if train_btn:
    X, y, features = preprocess_for_model(df, target_col="productivity_rating")

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    model = RandomForestRegressor(n_estimators=200, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    st.success("Model trained successfully! 🎯")
    st.metric("Model R² Score", f"{model.score(X_test, y_test):.2f}")

    st.subheader("Sample Predictions:")
    st.dataframe(pd.DataFrame({"Predicted Productivity": preds}).head())

st.markdown('</div>', unsafe_allow_html=True)
