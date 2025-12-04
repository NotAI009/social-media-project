from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import re

def extract_keywords_tfidf(text_series, topk=10):
    text_series = text_series.dropna().astype(str)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(text_series)
    scores = zip(vectorizer.get_feature_names_out(), tfidf.sum(axis=0).A1)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return sorted_scores[:topk]

def basic_sentiment_score(text):
    text = text.lower()
    positive = ["good", "productive", "focus", "helpful", "useful"]
    negative = ["bad", "waste", "distract", "timepass", "lazy"]
    score = 0
    for p in positive:
        if p in text:
            score += 1
    for n in negative:
        if n in text:
            score -= 1
    return score

def preprocess_for_model(df, target_col):
    cols = ["screen_time_hours", "study_hours", "sleep_hours"]
    X = df[cols].fillna(df[cols].median())
    y = df[target_col]
    return X, y, cols
st.caption("Dashboard created with ❤️ for academic research.")
