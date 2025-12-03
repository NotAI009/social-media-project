# app.py  (Full upgraded Streamlit app)
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
import base64
from datetime import datetime

# ✅ **2. Better App Title + Theme (copy into your app.py)**
# Custom Theme Styling
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 900;
    color: #00E5FF;
}
.subtitle {
    font-size: 18px;
    color: #B0BEC5;
    margin-bottom: 20px;
}
.section-header {
    font-size: 24px;
    font-weight: 700;
    color: #80DEEA;
    margin-top: 20px;
}
.block {
    background-color: #0A192F;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #1E3A8A;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Social Media vs Productivity (Demo)", layout="wide",
                   initial_sidebar_state="expanded")

# -------------------------
# Helper functions
# -------------------------
@st.cache_data
def load_data(path="sample_data.csv"):
    df = pd.read_csv(path)
    return df

def add_row(df, row_dict):
    return pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)

def linear_fit_line(x, y):
    model = LinearRegression().fit(x.reshape(-1,1), y)
    x_range = np.linspace(x.min(), x.max(), 100)
    y_pred = model.predict(x_range.reshape(-1,1))
    return x_range, y_pred, model

def bootstrap_confidence_interval(x, y, n_boot=2000, alpha=0.05):
    # bootstrap slope distribution for simple linear regression slope (study_hours ~ screen_time)
    slopes = []
    for i in range(n_boot):
        xi, yi = resample(x, y)
        m = LinearRegression().fit(xi.reshape(-1,1), yi)
        slopes.append(m.coef_[0])
    lower = np.percentile(slopes, 100*alpha/2)
    upper = np.percentile(slopes, 100*(1-alpha/2))
    return lower, upper, np.mean(slopes)

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

def download_link(object_to_download, download_filename, button_text):
    st.download_button(button_text, object_to_download, download_filename)

# -------------------------
# Load default dataset
# -------------------------
df = load_data("sample_data.csv")

# App header
st.markdown("""
<style>
.header-title{font-size:30px; font-weight:700; color:#03DAC6;}
.header-sub{color:#B0BEC5; margin-bottom:10px;}
.card {background-color:#0f1724; padding:14px; border-radius:8px;}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([8,2])
with col1:
    st.markdown("<div class='header-title'>Impact of Social Media on Student Productivity</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-sub'>Interactive demo — Upload your data or use the sample data. Live simulation, models and downloadable reports.</div>", unsafe_allow_html=True)
with col2:
    st.image("https://raw.githubusercontent.com/plotly/datasets/master/plotly-logo.png", width=80)

st.markdown("---")

# Sidebar controls & filters
with st.sidebar:
    st.header("Controls & Filters")
    year_options = ["All"] + sorted(df['year'].unique().tolist())
    selected_year = st.selectbox("Year", year_options, index=0)
    purpose_options = ["All"] + sorted(df['purpose'].unique().tolist())
    selected_purpose = st.selectbox("Purpose", purpose_options, index=0)
    focus_mode = st.selectbox("Uses focus mode", ["All", "yes", "no"], index=0)
    show_table = st.checkbox("Show data table", value=False)
    st.markdown("---")
    st.write("Model options")
    model_choice = st.selectbox("Model to predict study hours", ["Linear Regression", "Random Forest"], index=0)
    run_bootstrap = st.checkbox("Compute bootstrap CI (slower)", value=False)
    st.markdown("---")
    st.write("Project metadata")
    st.text_input("Group name", value="Group X", key="group")
    st.text_input("School/College", value="Your School", key="school")
    st.caption("Tip: Use the 'Add Entry' tab to insert a sample student live for demoing.")

# Apply filters to a working copy
df_display = df.copy()
if selected_year != "All":
    df_display = df_display[df_display['year'] == selected_year]
if selected_purpose != "All":
    df_display = df_display[df_display['purpose'] == selected_purpose]
if focus_mode != "All":
    df_display = df_display[df_display['uses_focus_mode'] == focus_mode]

# Main area tabs
tab_overview, tab_analysis, tab_simulator, tab_add = st.tabs(["Overview", "Analysis", "Simulator", "Add Entry"])

# Overview tab
with tab_overview:
    st.subheader("Quick summary")
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

    st.markdown("### Data (sample)")
    st.write("You can download the dataset or view it below.")
    download_link(df_to_csv_bytes(df_display), f"dataset_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "Download filtered dataset")
    if show_table:
        st.dataframe(df_display.reset_index(drop=True), use_container_width=True)

# Analysis tab
with tab_analysis:
    st.subheader("Screen time vs Study hours — analysis")
    # Scatter + regression line (sklearn)
    x = df_display['screen_time_hours'].to_numpy()
    y = df_display['study_hours_per_day'].to_numpy()

    if len(x) < 5:
        st.warning("Not enough data points for robust analysis. Add more entries or remove filters.")
    else:
        # Fit based on model choice
        if model_choice == "Linear Regression":
            model = LinearRegression().fit(x.reshape(-1,1), y)
            # compute bootstrap CI optionally
            if run_bootstrap:
                lower, upper, mean_slope = bootstrap_confidence_interval(x, y, n_boot=1000, alpha=0.05)
            else:
                lower, upper, mean_slope = (None, None, model.coef_[0])
            slope = model.coef_[0]
            intercept = model.intercept_
            y_pred = model.predict(x.reshape(-1,1))
            r2 = r2_score(y, y_pred)
            title = f"Linear Regression slope={slope:.3f}, R²={r2:.3f}"
        else:
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(x.reshape(-1,1), y)
            # approximate slope by partial dependence (difference over range)
            xr = np.linspace(x.min(), x.max(), 100)
            preds = rf.predict(xr.reshape(-1,1))
            slope = (preds[-1] - preds[0]) / (xr[-1] - xr[0])
            intercept = None
            r2 = r2_score(y, rf.predict(x.reshape(-1,1)))
            title = f"Random Forest (approx slope={slope:.3f}), R²={r2:.3f}"

        # scatter with regression line
        fig = px.scatter(df_display, x='screen_time_hours', y='study_hours_per_day', color='purpose',
                         hover_data=['gpa','sleep_hours'], title="Screen time vs Study hours")
        # overlay sklearn linear fit if linear model chosen (or add approximated line)
        xr = np.linspace(x.min(), x.max(), 100)
        if model_choice == "Linear Regression":
            fig.add_trace(go.Scatter(x=xr, y=intercept + slope*xr, mode='lines', line=dict(color='black', width=3),
                                     name='Linear fit'))
        else:
            fig.add_trace(go.Scatter(x=xr, y=rf.predict(xr.reshape(-1,1)), mode='lines', line=dict(color='black', width=3),
                                     name='RF fit'))

        st.plotly_chart(fig, use_container_width=True)

        # Stats box
        st.markdown("#### Key stats")
        st.write(f"Pearson correlation (screen_time vs study_hours): **{np.corrcoef(x,y)[0,1]:.3f}**")
        st.write(f"Model slope (hours study change per hour screen): **{slope:.3f}**")
        st.write(f"Model R²: **{r2:.3f}**")
        if run_bootstrap and lower is not None:
            st.write(f"Bootstrap slope CI (95%): [{lower:.3f}, {upper:.3f}]  — mean slope {mean_slope:.3f}")

    # allow download chart as png
    buf = io.BytesIO()
    fig.write_image(buf, format="png", width=1200, height=600, scale=2)
    buf.seek(0)
    st.download_button("Download chart (PNG)", buf, file_name="screen_vs_study.png", mime="image/png")

# Simulator tab
with tab_simulator:
    st.subheader("Simulation: What happens if average screen-time is reduced?")
    avg_screen = df_display['screen_time_hours'].mean()
    reduce_pct = st.slider("Reduce social media by (%)", 0, 100, 20)
    delta_hours = avg_screen * (reduce_pct/100.0)
    st.write(f"Average screen time in filtered sample: **{avg_screen:.2f} hrs/day**. Reducing by **{reduce_pct}%** = **{delta_hours:.2f} hrs/day**.")

    # Use the simple linear slope if available (from last analysis)
    if 'slope' in locals():
        predicted_change_hours = -slope * delta_hours  # note slope negative expected
        predicted_change_minutes = predicted_change_hours * 60
        st.write(f"Predicted mean study-hours change: **{predicted_change_hours:.2f} hours/day** (~**{predicted_change_minutes:.0f} minutes/day**).")
        # Simulate effect on GPA roughly (assume 0.1 GPA per 30 minutes extra study — illustrative)
        gpa_change = (predicted_change_minutes/30.0)*0.1
        st.write(f"Estimated GPA change (very rough estimate): **+{gpa_change:.2f}** (illustrative).")
    else:
        st.info("Not enough model info to run prediction. Run Analysis step first.")

    # scenario button: apply to dataset and show before/after hist
    if st.button("Show before/after study-hours distribution"):
        df_sim = df_display.copy()
        df_sim['sim_screen'] = df_sim['screen_time_hours'] * (1 - reduce_pct/100.0)
        if model_choice == "Linear Regression":
            df_sim['sim_study'] = intercept + slope * df_sim['sim_screen']
        else:
            df_sim['sim_study'] = rf.predict(df_sim['sim_screen'].values.reshape(-1,1))
        # clip to reasonable
        df_sim['sim_study'] = np.clip(df_sim['sim_study'], 0, 24)
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Histogram(x=df_display['study_hours_per_day'], name='Before', opacity=0.7))
        fig_sim.add_trace(go.Histogram(x=df_sim['sim_study'], name='After', opacity=0.7))
        fig_sim.update_layout(barmode='overlay', title="Before vs After study-hours distribution (simulated)")
        st.plotly_chart(fig_sim, use_container_width=True)
        # show download
        csv_bytes = df_sim.to_csv(index=False).encode('utf-8')
        st.download_button("Download simulated dataset (CSV)", csv_bytes, "simulated_dataset.csv")

# Add Entry tab (live demo)
with tab_add:
    st.subheader("Add a sample student (demo) — instantly updates charts")
    with st.form("add_form", clear_on_submit=True):
        a_age = st.number_input("Age", min_value=14, max_value=30, value=17)
        a_year = st.selectbox("Year", options=["11th","12th"])
        a_screen = st.number_input("Screen time (hrs/day)", min_value=0.0, max_value=24.0, value=3.0, step=0.25)
        a_purpose = st.selectbox("Purpose", options=["Entertainment","Study","Social","News","Other"])
        a_study = st.number_input("Study hours/day", min_value=0.0, max_value=24.0, value=3.5, step=0.25)
        a_gpa = st.number_input("GPA (1-10)", min_value=1.0, max_value=10.0, value=6.0, step=0.1)
        a_sleep = st.number_input("Sleep hours", min_value=0.0, max_value=24.0, value=7.0, step=0.25)
        a_apps = st.selectbox("Uses study apps", options=["yes","no"])
        a_focus = st.selectbox("Uses focus mode", options=["yes","no"])
        submitted = st.form_submit_button("Add this entry to dataset (demo only)")
    if submitted:
        new_row = {
            'id': df['id'].max() + 1 if 'id' in df.columns else len(df)+1,
            'age': a_age,
            'year': a_year,
            'screen_time_hours': a_screen,
            'purpose': a_purpose,
            'study_hours_per_day': a_study,
            'gpa': a_gpa,
            'sleep_hours': a_sleep,
            'uses_study_apps': a_apps,
            'uses_focus_mode': a_focus,
            'expected_gain_minutes_if_reduce_1h': np.nan
        }
        df = add_row(df, new_row)
        df_display = df.copy()
        st.success("Added — charts and analysis will update. (This change is in-memory only for the session.)")
        if show_table:
            st.dataframe(df_display.tail(5))
    st.info("Note: This adds rows only in your current session (not saved to disk). Use the 'Download filtered dataset' in Overview to export.")

st.markdown("---")
st.caption("Built with Streamlit — demo-ready. For deployment, Streamlit Community Cloud is easiest. If you need a ZIP of the project or a deployment guide, ask me.")
