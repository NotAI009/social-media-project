import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from utils import preprocess_for_model, extract_keywords_tfidf, basic_sentiment_score

st.set_page_config(page_title="Student Productivity Analysis Dashboard", layout="wide")

# ─────────────────────────────
# Custom CSS for styling + animations
# ─────────────────────────────
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

    .metric-card {
        background: #111827;
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #1f2937;
        animation: fadeInUp 0.7s ease-out;
        animation-fill-mode: both;
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
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
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

st.title("📊 Impact of Social Media Usage on Student Productivity")
st.markdown("""
This dashboard is based on a student survey and uses data analysis plus a Machine Learning model  
to study how **screen time**, **study hours**, **sleep** and **social media habits** relate to **productivity**.
""")

# ─────────────────────────────
# SIDEBAR: upload + train button
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
# LANDING SCREEN (before CSV upload)
# ─────────────────────────────
if uploaded is None:
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
    st.info(
        "To begin, upload the **Google Forms CSV** from the sidebar. "
        "Once the file is loaded, the full dashboard with tabs and charts will appear."
    )
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
# SIDEBAR FILTERS (for EDA only)
# ─────────────────────────────
st.sidebar.subheader("Filters (for EDA & text insights)")

gender_filter = None
purpose_filter = None

if "gender" in df.columns:
    gender_filter = st.sidebar.multiselect(
        "Filter by gender",
        options=sorted(df["gender"].dropna().unique().tolist()),
        default=sorted(df["gender"].dropna().unique().tolist())
    )

if "open_response" in df.columns:
    purpose_filter = st.sidebar.multiselect(
        "Filter by purpose",
        options=sorted(df["open_response"].dropna().unique().tolist()),
        default=sorted(df["open_response"].dropna().unique().tolist())
    )

# apply filters to create df_view (used for EDA & text)
df_view = df.copy()
if gender_filter:
    df_view = df_view[df_view["gender"].isin(gender_filter)]
if purpose_filter:
    df_view = df_view[df_view["open_response"].isin(purpose_filter)]

# ─────────────────────────────
# TABS  (ADDED: 📐 Math View)
# ─────────────────────────────
tab_overview, tab_eda, tab_text, tab_ml, tab_math = st.tabs(
    ["📋 Overview", "📊 EDA & Comparisons", "💬 Text Insights", "🤖 ML Model", "📐 Math View"]
)

# ========= OVERVIEW TAB =========
with tab_overview:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data Snapshot</div>', unsafe_allow_html=True)
    st.write(f"Total responses in file: **{len(df)}**")
    st.write(f"Responses after filters: **{len(df_view)}**")
    st.dataframe(df_view)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Key Summary Statistics (Filtered)</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label="Avg Screen Time", value=f"{df_view['screen_time_hours'].mean():.2f} hrs")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label="Avg Study Time", value=f"{df_view['study_hours'].mean():.2f} hrs")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label="Avg Sleep", value=f"{df_view['sleep_hours'].mean():.2f} hrs")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label="Avg Productivity", value=f"{df_view['productivity_rating'].mean():.2f} / 10")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========= EDA TAB =========
with tab_eda:
    # numeric correlations
    try:
        num_cols = ["screen_time_hours", "study_hours", "sleep_hours", "productivity_rating"]
        corr = df_view[num_cols].corr()
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Quick Correlations</div>', unsafe_allow_html=True)
        st.write("Correlation between study hours and productivity:",
                 f"**{corr.loc['study_hours','productivity_rating']:.2f}**")
        st.write("Correlation between screen time and productivity:",
                 f"**{corr.loc['screen_time_hours','productivity_rating']:.2f}**")
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception:
        pass

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
            title="Screen Time vs Productivity"
        )
        st.plotly_chart(fig1, use_container_width=True)
    with colB:
        fig2 = px.scatter(
            df_view,
            x="study_hours",
            y="productivity_rating",
            trendline="ols",
            trendline_color_override="red",
            title="Study Hours vs Productivity"
        )
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sleep & Group Comparisons</div>', unsafe_allow_html=True)
    fig3 = px.histogram(
        df_view,
        x="sleep_hours",
        nbins=20,
        title="Sleep Hours Distribution"
    )
    st.plotly_chart(fig3, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        if "gender" in df_view.columns:
            gender_counts = df_view["gender"].value_counts()
            fig_gender = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title="Gender Distribution"
            )
            st.plotly_chart(fig_gender, use_container_width=True)
    with colD:
        if "open_response" in df_view.columns:
            purpose_counts = df_view["open_response"].value_counts()
            fig_purpose_pie = px.pie(
                values=purpose_counts.values,
                names=purpose_counts.index,
                title="Purpose of Social Media Usage"
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
            title="Most Used Social Media Apps"
        )
        st.plotly_chart(fig_apps, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ========= TEXT TAB =========
with tab_text:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Keyword & Sentiment Analysis</div>', unsafe_allow_html=True)
    if "open_response" in df_view.columns:
        topk = st.slider("Select number of top keywords", 5, 30, 10)
        keywords = extract_keywords_tfidf(df_view["open_response"], topk)
        st.subheader("🔑 Top Keywords")
        st.table(pd.DataFrame(keywords, columns=["keyword", "score"]))
        df_view["sentiment_score"] = df_view["open_response"].apply(lambda x: basic_sentiment_score(str(x)))
        st.metric("Avg Sentiment Score", f"{df_view['sentiment_score'].mean():.2f}")
        fig_sent = px.histogram(
            df_view,
            x="sentiment_score",
            nbins=20,
            title="Sentiment Score Distribution"
        )
        st.plotly_chart(fig_sent, use_container_width=True)
    else:
        st.info("No open text column found for insights.")
    st.markdown('</div>', unsafe_allow_html=True)

# ========= ML TAB =========
with tab_ml:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Training & Predictions</div>', unsafe_allow_html=True)

    # Persist model across reruns
    if "model_data" not in st.session_state:
        st.session_state["model_data"] = None

    # Load model if user uploads one
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
            from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

            X, y, features = preprocess_for_model(df, target_col="productivity_rating")

            # Force regression for numeric rating
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
    else:
        st.info("Upload a model or click **Train Model From CSV** to enable predictions.")

    st.markdown('</div>', unsafe_allow_html=True)

# ========= 📐 MATH VIEW TAB (Matrices, Vectors, Integration) =========
with tab_math:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Matrix, Vector Algebra & Integration View</div>', unsafe_allow_html=True)
    st.write("""
    This tab explicitly shows how the project uses **matrices**, **vectors**,  
    and **integration**, matching the maths syllabus.
    """)

    # --- Feature Matrix using preprocess_for_model ---
    try:
        X_math, y_math, feature_cols_math = preprocess_for_model(df_view, target_col="productivity_rating")
        st.subheader("1️⃣ Feature Matrix (X)")
        st.write("Each row represents one student, each column is a feature (screen time, study hours, etc.).")
        st.latex(r"X = \begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1n} \\ x_{21} & x_{22} & \cdots & x_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ x_{m1} & x_{m2} & \cdots & x_{mn} \end{bmatrix}")
        st.dataframe(X_math)

        # --- Vector Algebra: choose one student and see feature vector ---
        st.subheader("2️⃣ Feature Vector for a Selected Student")
        if len(X_math) > 0:
            idx = st.number_input("Select student index (row in matrix):", 0, len(X_math) - 1, 0, 1)
            vec = X_math.iloc[int(idx)].values
            st.write("Selected feature vector:")
            st.latex(r"\vec{v} = \left[" + " \; ".join([str(round(v, 2)) for v in vec]) + r"\right]")
            st.write("This is the **feature vector** used by the ML model for that student.")
        else:
            st.info("Not enough data to form a feature matrix.")
    except Exception as e:
        st.warning(f"Could not form feature matrix: {e}")

    # --- Integration: area under productivity vs study_hours curve ---
    st.subheader("3️⃣ Integration: Productivity Curve (∫ Productivity vs Study Hours)")
    if "study_hours" in df_view.columns and "productivity_rating" in df_view.columns:
        df_int = df_view.dropna(subset=["study_hours", "productivity_rating"]).copy()
        if len(df_int) >= 2:
            df_int = df_int.sort_values("study_hours")
            x = df_int["study_hours"].values
            y = df_int["productivity_rating"].values

            # Trapezoidal numerical integration
            area = np.trapz(y, x)
            st.write("""
            We approximate the area under the **Productivity vs Study Hours** curve using the  
            **trapezoidal rule**, which is a standard numerical integration technique:
            """)
            st.latex(r"\text{Productivity Index} \approx \int f(t)\,dt \approx \sum \frac{(x_{i+1}-x_i)\,[f(x_i)+f(x_{i+1})]}{2}")
            st.metric("Integration-based Productivity Index", f"{area:.2f}")

            # Plot the curve
            fig_int = px.area(df_int, x="study_hours", y="productivity_rating",
                              title="Area under Productivity vs Study Hours curve (Integration)")
            st.plotly_chart(fig_int, use_container_width=True)
        else:
            st.info("Not enough distinct data points to perform integration.")
    else:
        st.info("Need 'study_hours' and 'productivity_rating' columns to demonstrate integration.")

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────
# FOOTER
# ─────────────────────────────
st.markdown("---")
st.caption("Dashboard created with ❤️ for academic research.")
