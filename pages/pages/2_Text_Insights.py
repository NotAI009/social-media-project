# pages/2_Text_Insights.py

import streamlit as st
import pandas as pd
import plotly.express as px
from utils import extract_keywords_tfidf, basic_sentiment_score

st.set_page_config(page_title="Text Insights · Student Productivity", layout="wide")

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

st.title("💬 Text Insights")

df = st.session_state.get("df", None)
if df is None:
    st.error("No data found. Go to the **Home** page and upload the Google Forms CSV first.")
    st.stop()

# Sidebar filters (same logic)
st.sidebar.header("Filters")

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
        "Filter by purpose of use",
        options=sorted(df["open_response"].dropna().unique().tolist()),
        default=sorted(df["open_response"].dropna().unique().tolist()),
    )

df_view = df.copy()
if gender_filter:
    df_view = df_view[df_view["gender"].isin(gender_filter)]
if purpose_filter:
    df_view = df_view[df_view["open_response"].isin(purpose_filter)]

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Keyword & Sentiment Analysis</div>', unsafe_allow_html=True)

if "open_response" in df_view.columns:
    topk = st.slider("Select number of top keywords", 5, 30, 10)
    keywords = extract_keywords_tfidf(df_view["open_response"], topk)
    st.subheader("🔑 Top Keywords")
    st.table(pd.DataFrame(keywords, columns=["keyword", "score"]))

    df_view["sentiment_score"] = df_view["open_response"].apply(lambda x: basic_sentiment_score(str(x)))
    st.metric("Average Sentiment Score", f"{df_view['sentiment_score'].mean():.2f}")

    fig_sent = px.histogram(
        df_view,
        x="sentiment_score",
        nbins=20,
        title="Sentiment Score Distribution",
    )
    st.plotly_chart(fig_sent, use_container_width=True)
else:
    st.info("No open text response column available.")

st.markdown('</div>', unsafe_allow_html=True)
