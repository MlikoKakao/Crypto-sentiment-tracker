from src.domain.sentiment.service import add_sentiment_to_df
import pandas as pd
import streamlit as st
from typing import Dict, Any, List
from src.domain.sentiment.registry import ALL_ANALYZER_NAMES, ANALYZERS

def analyze_benchmark_posts() -> None:
    raw_posts_df = pd.read_csv("data/benchmark/benchmark_labeled.csv")
    
    for analyzer in ALL_ANALYZER_NAMES:
        df = add_sentiment_to_df(raw_posts_df, analyzer)
        df["label_pred"] = "neutral"
        
        df.to_csv(f"data/benchmark/scored/{analyzer}.csv", index=False)

def load_benchmark_csv() -> Dict[str, pd.DataFrame]:
    loaded: Dict[str, pd.DataFrame] = {}
    for analyzer in ALL_ANALYZER_NAMES:
        loaded[analyzer] = pd.read_csv(f"data/benchmark/scored/{analyzer}.csv")
    
    return loaded

def convert_score_to_tri(score: float) -> str:
    if score >= 0.05: return "positive"
    elif score <= -0.05: return "negative"
    else: return "neutral"

def list_sentiment(analyzer: str, text: List[str]) -> List[str]:
    scored_list = ANALYZERS[analyzer](text)

def show_benchmark_data():
    sentiment_dfs = load_benchmark_csv()
    results: Dict[str, Dict[str, Any]] = {}