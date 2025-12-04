# pages/1_Dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard · Student Productivity", layout="wide")

# CSS reused
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
    .metric-card, .section-card {
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

st.title("📋 Dashboard")

# ---------- Get data from session_state ----------
df = st.session_state.get("df", None)
if df is None:
    st.error("No data found. Go to the **Home** page and upload the Google Forms CSV first.")
    st.stop()

# ---------- Sidebar filters ----------
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

# ---------- Data snapshot ----------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Data Snapshot</div>', unsafe_allow_html=True)
st.write(f"Total responses in file: **{len(df)}**")
st.write(f"Responses after filters: **{len(df_view)}**")
st.dataframe(df_view)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Summary metrics ----------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Key Summary Statistics (Filtered)</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Avg Screen Time", f"{df_view['screen_time_hours'].mean():.2f} hrs")
with col2:
    st.metric("Avg Study Time", f"{df_view['study_hours'].mean():.2f} hrs")
with col3:
    st.metric("Avg Sleep", f"{df_view['sleep_hours'].mean():.2f} hrs")
with col4:
    st.metric("Avg Productivity", f"{df_view['productivity_rating'].mean():.2f} / 10")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Quick correlations ----------
try:
    num_cols = ["screen_time_hours", "study_hours", "sleep_hours", "productivity_rating"]
    corr = df_view[num_cols].corr()
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Quick Correlations</div>', unsafe_allow_html=True)
    st.write("Study hours vs productivity:", f"**{corr.loc['study_hours','productivity_rating']:.2f}**")
    st.write("Screen time vs productivity:", f"**{corr.loc['screen_time_hours','productivity_rating']:.2f}**")
    st.markdown('</div>', unsafe_allow_html=True)
except Exception:
    pass

# ---------- Scatter plots with trendlines ----------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Screen Time & Study vs Productivity</div>', unsafe_allow_html=True)
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
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Sleep + distributions ----------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Sleep & Group Comparisons</div>', unsafe_allow_html=True)

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
