import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Features used by the model (no GPA, no name/gender/etc.)
FEATURE_HELPER_COLS = [
    "screen_time_hours",
    "study_hours",
    "sleep_hours",
    "screen_per_study"
]

def preprocess_for_model(df, target_col="productivity_rating"):
    """
    Prepares dataset for ML model training.
    Handles numeric conversion, missing values, and derived features.
    """
    df = df.copy()
    df = df.dropna(subset=[target_col])

    # Numeric conversions
    for c in ["screen_time_hours", "study_hours", "sleep_hours"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived feature
    if "screen_time_hours" in df.columns and "study_hours" in df.columns:
        df["screen_per_study"] = df["screen_time_hours"] / (df["study_hours"] + 0.1)

    # Select features that exist
    features = [c for c in FEATURE_HELPER_COLS if c in df.columns]

    X = df[features].fillna(df[features].median())
    y = df[target_col]

    # Convert numeric-looking strings
    try:
        y = pd.to_numeric(y, errors="ignore")
    except:
        pass

    return X, y, features


# -----------------------------
# TEXT ANALYTICS HELPERS
# -----------------------------
def extract_keywords_tfidf(text_series, topk=10):
    text_series = text_series.fillna("").astype(str)
    tf = TfidfVectorizer(stop_words="english", max_features=200)
    mat = tf.fit_transform(text_series)
    scores = mat.sum(axis=0).A1
    terms = tf.get_feature_names_out()

    pairs = list(zip(terms, scores))
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    return pairs[:topk]


def basic_sentiment_score(text):
    """
    Very simple sentiment scoring using keyword sets.
    Positive words add +1, negative words add -1.
    """
    pos = {
        "productive", "focused", "motivated", "study", "learning",
        "progress", "efficient", "improving"
    }

    neg = {
        "distracted", "waste", "addicted", "tired",
        "procrastinate", "procrastination", "overuse", "unproductive"
    }

    cleaned = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    words = cleaned.split()

    score = 0
    for w in words:
        if w in pos:
            score += 1
        if w in neg:
            score -= 1

    return score
