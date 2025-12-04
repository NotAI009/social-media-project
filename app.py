# app.py  (HOME / landing page)

import streamlit as st
import pandas as pd
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Student Productivity Dashboard", layout="wide")

# ---------- CSS + basic animations ----------
st.markdown("""
    <style>
    .main {
        background-color: #050816;
        color: #eaeaea;
    }
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

    .hero-card {
        background: radial-gradient(circle at top left, #1d4ed8, #0b1120 55%);
        padding: 2.5rem 3rem;
        border-radius: 1.5rem;
        border: 1px solid #1f2937;
        color: #e5e7eb;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
        animation-fill-mode: both;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.75rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.95;
        margin-bottom: 1.25rem;
    }
    .hero-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.5);
        font-size: 0.8rem;
        margin-bottom: 0.75rem;
    }
    .hero-list li {
        margin-bottom: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)


def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


# free analytics animation from lottiefiles
LOTTIE_URL = "https://assets2.lottiefiles.com/packages/lf20_touohxv0.json"
lottie_anim = load_lottie_url(LOTTIE_URL)

st.title("📊 Impact of Social Media Usage on Student Productivity")
st.markdown("""
This dashboard is based on a student survey and uses data analysis plus a Machine Learning model  
to study how **screen time**, **study hours**, **sleep** and **social media habits** relate to **productivity**.
""")

# ---------- Sidebar: CSV upload ----------
with st.sidebar:
    st.header("Upload survey data")
    uploaded = st.file_uploader("Upload Google Forms CSV", type=["csv"])

    st.markdown("---")
    st.caption("After uploading, open the **Dashboard** page from the navigation bar at the top.")

# ---------- If file uploaded, clean + store in session_state ----------
if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    df_raw.columns = df_raw.columns.str.strip()

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
    df = df_raw.rename(columns=col_map)

    st.session_state["df"] = df  # make it available to all pages

    st.success("CSV loaded and columns mapped successfully ✅")
    st.info("Open the **Dashboard** page (top navigation) to explore the data.")
else:
    st.info("Upload the **Google Forms CSV** from the sidebar to start the analysis.")

# ---------- Hero content with Lottie ----------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">Step 1 · Upload the survey CSV using the panel on the left</div>
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
    if lottie_anim is not None:
        st_lottie(lottie_anim, height=260, key="analytics")
    else:
        st.empty()

st.markdown("---")
st.caption("Use the navigation bar at the top to open **Dashboard**, **Text Insights**, and **ML Model** pages.")
