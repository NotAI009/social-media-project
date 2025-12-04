import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------------------------------------------------------
# 1. CONFIG & STYLING (The "Way Better" CSS)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Student Productivity Hub", page_icon="🎓", layout="wide")

def inject_custom_css():
    st.markdown("""
        <style>
        /* Main background and font */
        .stApp {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', sans-serif;
        }
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #2c3e50;
        }
        /* Custom Cards for Metrics */
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        /* Headings */
        h1, h2, h3 {
            color: #2c3e50;
        }
        /* Navigation Radio Button in Sidebar */
        .stRadio > label {
            color: white !important;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS (Formerly utils.py)
# -----------------------------------------------------------------------------
def clean_data(df):
    col_map = {
        "4. Average daily social media screen time (in hours)": "screen_time_hours",
        "5. Average daily study hours (in hours)": "study_hours",
        "6. Average daily sleep hours": "sleep_hours",
        "10. Productivity Rating (1–10)": "productivity_rating",
        "8. Purpose of social media usage": "open_response",
        "7. Top social media apps you use": "social_apps",
        "3. Gender": "gender",
        "1. NAME": "name"
    }
    # Fuzzy rename
    df = df.rename(columns=lambda x: col_map.get(x.strip(), x.strip()))
    
    # Numeric conversion
    for col in ['screen_time_hours', 'study_hours', 'sleep_hours', 'productivity_rating']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # Feature Engineering
    if "screen_time_hours" in df.columns and "study_hours" in df.columns:
        df["screen_per_study"] = df["screen_time_hours"] / (df["study_hours"] + 0.1)
        
    return df

def preprocess_for_model(df, target_col):
    feature_cols = ["screen_time_hours", "study_hours", "sleep_hours", "screen_per_study"]
    if "gender" in df.columns:
        df = pd.get_dummies(df, columns=["gender"], drop_first=True)
        feature_cols.extend([c for c in df.columns if "gender_" in c])
    
    available = [f for f in feature_cols if f in df.columns]
    return df[available].fillna(0), df[target_col], available

def get_text_insights(text_series, top_n=10):
    text_data = text_series.dropna().astype(str).tolist()
    if not text_data: return [], []
    
    # Keywords
    vec = TfidfVectorizer(stop_words='english', max_features=top_n)
    vec.fit(text_data)
    indices = np.argsort(vec.idf_)[::-1]
    features = vec.get_feature_names_out()
    keywords = [(features[i], vec.idf_[i]) for i in indices[:top_n]]
    
    # Sentiment
    sentiments = [TextBlob(t).sentiment.polarity for t in text_data]
    return keywords, sentiments

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & UPLOAD
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎓 Student Hub")
    
    # Navigation Menu
    menu = st.radio("Navigate", ["🏠 Home", "📊 Analytics", "🤖 Prediction Lab", "💬 Text Insights"])
    
    st.markdown("---")
    st.subheader("📂 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    # Load Data Logic
    if "df" not in st.session_state:
        st.session_state["df"] = None

    if uploaded_file:
        raw = pd.read_csv(uploaded_file)
        st.session_state["df"] = clean_data(raw)
        st.success("Data Loaded!")
    
    st.info("Developed for Academic Research")

# -----------------------------------------------------------------------------
# 4. PAGE: HOME
# -----------------------------------------------------------------------------
if menu == "🏠 Home":
    st.title("🎓 Student Productivity Analysis Hub")
    st.markdown("### Welcome to the Intelligence System")
    st.write("This dashboard analyzes the impact of social media, sleep, and study habits on productivity.")
    
    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        st.success(f"✅ Active Dataset: {len(df)} records loaded.")
        
        # Summary Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Productivity", f"{df['productivity_rating'].mean():.1f}/10")
        c2.metric("Avg Screen Time", f"{df['screen_time_hours'].mean():.1f} hrs")
        c3.metric("Avg Study Time", f"{df['study_hours'].mean():.1f} hrs")
        
        with st.expander("📄 View Raw Data"):
            st.dataframe(df.head())
    else:
        st.warning("👈 Please upload your CSV file in the sidebar to begin.")
        
        # Demo Data Button
        if st.button("Load Demo Data (Test Mode)"):
            data = {
                '4. Average daily social media screen time (in hours)': np.random.randint(1, 10, 50),
                '5. Average daily study hours (in hours)': np.random.randint(1, 8, 50),
                '6. Average daily sleep hours': np.random.randint(4, 9, 50),
                '10. Productivity Rating (1–10)': np.random.randint(1, 11, 50),
                '3. Gender': np.random.choice(['Male', 'Female'], 50),
                '8. Purpose of social media usage': np.random.choice(['News', 'Memes', 'Chat', 'Work'], 50),
                '7. Top social media apps you use': np.random.choice(['TikTok', 'Instagram', 'LinkedIn'], 50)
            }
            st.session_state["df"] = clean_data(pd.DataFrame(data))
            st.rerun()

# -----------------------------------------------------------------------------
# 5. PAGE: ANALYTICS
# -----------------------------------------------------------------------------
elif menu == "📊 Analytics":
    st.title("📊 Analytics Dashboard")
    
    if st.session_state["df"] is None:
        st.error("Please upload data first!")
    else:
        df = st.session_state["df"]
        
        # Tabs for cleaner UI
        tab1, tab2 = st.tabs(["📈 Correlations", "📱 Apps & Habits"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.scatter(df, x="screen_time_hours", y="productivity_rating", size="study_hours", 
                                color="gender" if "gender" in df.columns else None,
                                title="Impact of Screen Time on Productivity (Size = Study Hours)")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                corr = df[["screen_time_hours", "study_hours", "sleep_hours", "productivity_rating"]].corr()
                fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Heatmap")
                st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab2:
            if "social_apps" in df.columns:
                app_counts = df["social_apps"].astype(str).str.split(",").explode().str.strip().value_counts().reset_index()
                app_counts.columns = ["App", "Count"]
                fig_bar = px.bar(app_counts.head(8), x="Count", y="App", orientation='h', title="Top Apps Used", color="Count")
                st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. PAGE: PREDICTION LAB
# -----------------------------------------------------------------------------
elif menu == "🤖 Prediction Lab":
    st.title("🤖 AI Prediction Lab")
    
    if st.session_state["df"] is None:
        st.error("Please upload data first!")
    else:
        df = st.session_state["df"]
        
        # Train Button
        col_train, col_status = st.columns([1, 3])
        with col_train:
            train_btn = st.button("🚀 Train Model", type="primary")
        
        if "model" not in st.session_state:
            st.session_state["model"] = None
            
        if train_btn:
            X, y, features = preprocess_for_model(df, "productivity_rating")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            pipe = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestRegressor(n_estimators=100))])
            pipe.fit(X_train, y_train)
            
            st.session_state["model"] = pipe
            st.session_state["features"] = features
            acc = r2_score(y_test, pipe.predict(X_test))
            st.success(f"Model Trained! Accuracy (R²): {acc:.2%}")
            
        if st.session_state["model"]:
            st.markdown("---")
            st.subheader("🔮 Productivity Simulator")
            st.write("Adjust sliders to see how habits change predicted productivity.")
            
            c1, c2, c3 = st.columns(3)
            s_time = c1.slider("Social Media (Hrs)", 0.0, 12.0, 4.0)
            study = c2.slider("Study (Hrs)", 0.0, 12.0, 2.0)
            sleep = c3.slider("Sleep (Hrs)", 0.0, 12.0, 7.0)
            
            # Prepare Input
            input_data = pd.DataFrame({
                "screen_time_hours": [s_time], "study_hours": [study], "sleep_hours": [sleep],
                "screen_per_study": [s_time/(study+0.1)]
            })
            
            # Add dummy cols
            for f in st.session_state["features"]:
                if f not in input_data.columns: input_data[f] = 0
            
            pred = st.session_state["model"].predict(input_data[st.session_state["features"]])[0]
            
            st.markdown(f"""
                <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3>Predicted Productivity</h3>
                    <h1 style="color: #2e7d32; font-size: 50px;">{pred:.1f} / 10</h1>
                </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. PAGE: TEXT INSIGHTS
# -----------------------------------------------------------------------------
elif menu == "💬 Text Insights":
    st.title("💬 Qualitative Analysis")
    
    if st.session_state["df"] is None:
        st.error("Upload data first.")
    elif "open_response" not in st.session_state["df"].columns:
        st.warning("No open-text column found in dataset.")
    else:
        df = st.session_state["df"]
        keywords, sentiments = get_text_insights(df["open_response"])
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Sentiment Distribution")
            df["sentiment"] = pd.Series(sentiments)
            fig_hist = px.histogram(df, x="sentiment", nbins=15, title="Response Sentiment (-1 to +1)")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with c2:
            st.subheader("Top Keywords")
            kw_df = pd.DataFrame(keywords, columns=["Term", "Score"])
            fig_bar = px.bar(kw_df, x="Score", y="Term", orientation='h', color="Score")
            st.plotly_chart(fig_bar, use_container_width=True)
