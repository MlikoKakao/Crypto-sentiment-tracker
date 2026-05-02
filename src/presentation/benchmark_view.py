from src.domain.sentiment.service import add_sentiment_to_df
import pandas as pd
import streamlit as st
from typing import Dict, Any, List
from pathlib import Path
from src.benchmark.analyzer_eval import metrics
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
        if not Path(f"data/benchmark/scored/{analyzer}.csv").exists():
            analyze_benchmark_posts()
        loaded[analyzer] = pd.read_csv(f"data/benchmark/scored/{analyzer}.csv")
    
    return loaded

def convert_score_to_tri(score: float) -> str:
    if score >= 0.05: return "positive"
    elif score <= -0.05: return "negative"
    else: return "neutral"


def list_sentiment_to_tri(analyzer: str) -> List[str] | None:
    df = pd.read_csv(f"data/benchmark/scored/{analyzer}")
    if not df["sentiment"] in df.columns:
        st.warning("Benchmark CSV doesn't have required columns")
        return None;
    scores = df["sentiment"].tolist()
    for score in scores:
        scores = convert_score_to_tri(score) 
    return list(str(scores))



def show_benchmark_data():
    # sentiment_dfs = multiple dfs of each analyzer on benchmark dataset
    sentiment_dfs = load_benchmark_csv()
    for df in sentiment_dfs:
        y = list_sentiment_to_tri(df)

    results: Dict[str, Dict[str, Any]] = {}


    results["VADER"] = 