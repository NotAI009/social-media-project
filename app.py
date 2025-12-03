# app.py - Upgraded Streamlit app for "Impact of Social Media on Student Productivity"
# Features:
# - Polished UI and theme
# - Sidebar filters and model selection
# - Tabs: Overview, Analysis, Simulator, Add Entry
# - Robust plotting with sklearn regression or RandomForest
# - Bootstrap CI option
# - Live row entry (in-session)
# - Download dataset and chart (PNG with kaleido fallback to HTML)
# - Defensive code to avoid crashes on servers without image engines
# - Clean header and project metadata

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime

st.set_page_config(page_title="Social Media & Productivity — Interactive Demo",
                   layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Styling / Header
# -------------------------
st.markdown("""
<style>
body {background-color: #071427;}
.header-title {font-size:36px; font-weight:800; color:#00E5FF; margin-bottom:0;}
.header-sub {font-size:14px; color:#B0BEC5; margin-top:2px; margin-bottom:12px;}
.card {background-color:#071427; padding:12px; border-radius:10px; border:1px solid #0f1724;}
.sidebar .stSlider > div > div {color:#fff;}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([8,2])
with col1:
    st.markdown("<div class='header-title'>📱 Impact of Social Media on Student Productivity</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-sub'>Interactive dashboard — analyze, simulate, and demonstrate how screen-time affects study-hours and performance.</div>", unsafe_allow_html=True)
with col2:
    st.write("")  # reserved for image or logo

st.markdown("---")

# -------------------------
# Utilities
# -------------------------
@st.cache_data
def load_sample(path="sample_data.csv"):
    try:
        return pd.read_csv(path)
    except Exception:
        # fallback small sample
        return pd.DataFrame({
            'id':[1,2,3],
            'age':[17,18,16],
            'year':['11th','12th','11th'],
            'screen_time_hours':[3.2,4.5,1.5],
            'purpose':['Entertainment','Social','Study'],
            'study_hours_per_day':[3.5,2.0,5.0],
            'gpa':[6.5,5.2,8.0],
            'sleep_hours':[6.8,6.0,8.0],
            'uses_study_apps':['no','no','yes'],
            'uses_focus_mode':['no','yes','no'],
            'expected_gain_minutes_if_reduce_1h':[15,8,25]
        })

def download_bytes(obj_bytes, filename, mime):
    st.download_button(label=f"Download {filename}", data=obj_bytes, file_name=filename, mime=mime)

def add_row(df, row):
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

def bootstrap_slope_ci(x, y, n_boot=1000, alpha=0.05):
    slopes = []
    for i in range(n_boot):
        xi, yi = resample(x, y)
        m = LinearRegression().fit(xi.reshape(-1,1), yi)
        slopes.append(m.coef_[0])
    lower = np.percentile(slopes, 100*alpha/2)
    upper = np.percentile(slopes, 100*(1-alpha/2))
    return lower, upper, np.mean(slopes)

# -------------------------
# Load data
# -------------------------
df = load_sample("sample_data.csv")

# Sidebar controls
with st.sidebar:
    st.header("Controls & Filters")
    year_opts = ["All"] + sorted(df['year'].unique().tolist())
    selected_year = st.selectbox("Year", year_opts, index=0)
    purpose_opts = ["All"] + sorted(df['purpose'].unique().tolist())
    selected_purpose = st.selectbox("Purpose", purpose_opts, index=0)
    focus_mode = st.selectbox("Uses focus mode", ["All","yes","no"], index=0)
    show_table = st.checkbox("Show data table", value=False)
    st.markdown("---")
    st.write("Model & analysis options")
    model_choice = st.selectbox("Model", ["Linear Regression","Random Forest"], index=0)
    run_boot = st.checkbox("Bootstrap slope CI (slower)", value=False)
    st.markdown("---")
    st.write("Project info")
    st.text_input("Group / Author", value="Your Name Here", key="author")
    st.text_input("Institution", value="Your School", key="institution")
    st.markdown("Tip: Use the 'Add Entry' tab to add demo rows live.")

# Apply filters
df_display = df.copy()
if selected_year != "All":
    df_display = df_display[df_display['year'] == selected_year]
if selected_purpose != "All":
    df_display = df_display[df_display['purpose'] == selected_purpose]
if focus_mode != "All":
    df_display = df_display[df_display['uses_focus_mode'] == focus_mode]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Analysis", "Simulator", "Add Entry"])

# ---------- Overview ----------
with tab1:
    st.subheader("Overview & key metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples (N)", len(df_display))
    c2.metric("Median screen time (hrs)", f"{df_display['screen_time_hours'].median():.2f}")
    c3.metric("Median study hrs/day", f"{df_display['study_hours_per_day'].median():.2f}")
    c4.metric("Median GPA", f"{df_display['gpa'].median():.2f}")

    st.markdown("### Distributions")
    g1, g2 = st.columns([2,1])
    with g1:
        fig_hist = px.histogram(df_display, x='screen_time_hours', nbins=8, title="Screen time (hrs/day) distribution")
        st.plotly_chart(fig_hist, use_container_width=True)
    with g2:
        fig_pie = px.pie(df_display, names='purpose', title="Purpose of social media")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### Dataset")
    if show_table:
        st.dataframe(df_display.reset_index(drop=True), use_container_width=True)
    csv_bytes = df_display.to_csv(index=False).encode('utf-8')
    download_bytes(csv_bytes, f"dataset_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

# ---------- Analysis ----------
with tab2:
    st.subheader("Analysis: Screen time vs Study hours")
    if len(df_display) < 4:
        st.warning("Not enough data points to run reliable models. Add more rows or remove filters.")
    else:
        x = df_display['screen_time_hours'].to_numpy()
        y = df_display['study_hours_per_day'].to_numpy()

        if model_choice == "Linear Regression":
            model = LinearRegression().fit(x.reshape(-1,1), y)
            slope = model.coef_[0]
            intercept = model.intercept_
            preds = model.predict(x.reshape(-1,1))
        else:
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(x.reshape(-1,1), y)
            xr = np.linspace(x.min(), x.max(), 100)
            slope = (rf.predict(xr.reshape(-1,1))[-1] - rf.predict(xr.reshape(-1,1))[0]) / (xr[-1]-xr[0])
            intercept = None
            preds = rf.predict(x.reshape(-1,1))

        r2 = r2_score(y, preds)

        # Plot
        fig = px.scatter(df_display, x='screen_time_hours', y='study_hours_per_day', color='purpose',
                         hover_data=['gpa','sleep_hours'], title="Screen time vs Study hours")
        xr_line = np.linspace(x.min(), x.max(), 100)
        if model_choice == "Linear Regression":
            fig.add_trace(go.Scatter(x=xr_line, y=intercept + slope*xr_line, mode='lines', line=dict(color='black', width=3),
                                     name='Linear fit'))
        else:
            fig.add_trace(go.Scatter(x=xr_line, y=rf.predict(xr_line.reshape(-1,1)), mode='lines', line=dict(color='black', width=3),
                                     name='RF fit'))

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Key stats")
        st.write(f"Pearson correlation: **{np.corrcoef(x,y)[0,1]:.3f}**")
        st.write(f"Model slope (hours study per hour screen): **{slope:.3f}**")
        st.write(f"Model R²: **{r2:.3f}**")

        if run_boot:
            lower, upper, mean_slope = bootstrap_slope_ci(x, y, n_boot=1000, alpha=0.05)
            st.write(f"Bootstrap 95% CI for slope: [{lower:.3f}, {upper:.3f}]  (mean slope ≈ {mean_slope:.3f})")

        # Download chart — robust handling (kaleido may be missing on cloud)
        try:
            buf = io.BytesIO()
            fig.write_image(buf, format="png", width=1200, height=600, scale=2)
            buf.seek(0)
            download_bytes(buf.getvalue(), "screen_vs_study.png", "image/png")
        except Exception as e:
            # fallback to HTML
            st.warning("PNG export unavailable in this environment. Downloading interactive HTML instead.")
            html_bytes = fig.to_html(full_html=False, include_plotlyjs='cdn').encode('utf-8')
            download_bytes(html_bytes, "screen_vs_study.html", "text/html")

# ---------- Simulator ----------
with tab3:
    st.subheader("Simulator: Estimate effect of reducing screen time")
    avg_screen = df_display['screen_time_hours'].mean()
    reduce_pct = st.slider("Reduce social media by (%)", 0, 100, 20)
    delta = avg_screen * (reduce_pct/100.0)
    st.write(f"Average screen time: **{avg_screen:.2f} hrs/day**. Reducing by {reduce_pct}% = **{delta:.2f} hrs/day**.")

    # Use slope from previous model when available
    if 'slope' in locals():
        predicted_change_hours = -slope * delta
        st.write(f"Predicted mean study-hours change: **{predicted_change_hours:.2f} hours/day** (~**{predicted_change_hours*60:.0f} minutes/day**).")
        # estimate GPA delta (illustrative)
        gpa_change = (predicted_change_hours*60)/30 * 0.1
        st.write(f"Estimated GPA change (illustrative): **{gpa_change:.2f}**")
    else:
        st.info("Run analysis with enough data to enable predictions.")

    if st.button("Show before vs after distribution"):
        df_sim = df_display.copy()
        df_sim['sim_screen'] = df_sim['screen_time_hours'] * (1 - reduce_pct/100.0)
        if 'intercept' in locals() and intercept is not None:
            df_sim['sim_study'] = intercept + slope * df_sim['sim_screen']
        else:
            # fallback to linear approx using slope only
            df_sim['sim_study'] = df_sim['study_hours_per_day'] + (-slope * (df_sim['screen_time_hours'] - df_sim['sim_screen']))
        df_sim['sim_study'] = np.clip(df_sim['sim_study'], 0, 24)
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Histogram(x=df_display['study_hours_per_day'], name='Before', opacity=0.7))
        fig_sim.add_trace(go.Histogram(x=df_sim['sim_study'], name='After', opacity=0.7))
        fig_sim.update_layout(barmode='overlay', title="Before vs After study-hours (simulated)")
        st.plotly_chart(fig_sim, use_container_width=True)
        csv_bytes = df_sim.to_csv(index=False).encode('utf-8')
        download_bytes(csv_bytes, "simulated_dataset.csv", "text/csv")

# ---------- Add Entry (live) ----------
with tab4:
    st.subheader("Add a new survey entry (demo only)")
    with st.form("entry_form", clear_on_submit=True):
        a_age = st.number_input("Age", 14, 30, 17)
        a_year = st.selectbox("Year", ["11th","12th"])
        a_screen = st.number_input("Screen time (hrs/day)", 0.0, 24.0, 3.0, 0.25)
        a_purpose = st.selectbox("Purpose", ["Entertainment","Study","Social","News","Other"])
        a_study = st.number_input("Study hours/day", 0.0, 24.0, 3.5, 0.25)
        a_gpa = st.number_input("GPA (1-10)", 1.0, 10.0, 6.0, 0.1)
        a_sleep = st.number_input("Sleep hours", 0.0, 24.0, 7.0, 0.25)
        a_apps = st.selectbox("Uses study apps", ["yes","no"])
        a_focus = st.selectbox("Uses focus mode", ["yes","no"])
        submitted = st.form_submit_button("Add entry")
    if submitted:
        new_row = {
            'id': int(df['id'].max() + 1) if 'id' in df.columns else len(df)+1,
            'age': int(a_age),
            'year': a_year,
            'screen_time_hours': float(a_screen),
            'purpose': a_purpose,
            'study_hours_per_day': float(a_study),
            'gpa': float(a_gpa),
            'sleep_hours': float(a_sleep),
            'uses_study_apps': a_apps,
            'uses_focus_mode': a_focus,
            'expected_gain_minutes_if_reduce_1h': np.nan
        }
        df = add_row(df, new_row)
        df_display = df.copy()
        st.success("Added new entry (session-only). Charts and analysis update in this session.")
        if show_table:
            st.dataframe(df_display.tail(5))

st.markdown("---")
st.caption("Built with Streamlit. Deploy on Streamlit Cloud for a shareable URL (no installs required).")
