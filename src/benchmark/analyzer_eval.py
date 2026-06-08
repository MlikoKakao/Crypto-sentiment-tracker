from __future__ import annotations
import time
from typing import List, Dict, Any, Sequence, Tuple

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


from .benchmark_plot import to_table
from src.domain.sentiment.service import add_sentiment_to_df
from src.domain.sentiment.registry import ALL_ANALYZER_NAMES

CANONICAL = ("negative", "neutral", "positive")

def _to_trinary_from_score(x: float, pos: float = 0.05, neg: float = -0.05) -> str:
    if x >= pos: return "positive"
    if x <= neg: return "negative"
    return "neutral"


def metrics(true_label: List[str], prediction: List[str]) -> Dict[str, Any]:
    acc = accuracy_score(true_label, prediction)
    f1m = f1_score(true_label, prediction, average="macro")
    cm  = confusion_matrix(true_label, prediction, labels=list(CANONICAL))
    report = classification_report(true_label, prediction, labels=list(CANONICAL), zero_division=0, digits=3)
    return {"accuracy": acc, "f1_macro": f1m, "confusion": cm, "report": report}


def _examples(y_true: Sequence[str], y_pred: Sequence[str], texts: Sequence[str], k: int = 4) -> List[Tuple[str, str, str]]:
    bad = [(t, yt, yp) for t, yt, yp in zip(texts, y_true, y_pred) if yt != yp]
    return bad[:k]


def evaluate(df: pd.DataFrame,
             text_col: str = "text",
             label_col: str = "label",
             ) -> Dict[str, Dict[str, Any]]:
    # schema check (clear error if CSV is wrong)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in labeled CSV: {sorted(missing)}")

    df = df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str)

    labels_norm = df[label_col]
    # can add custom benchmarking later?
    y_true = labels_norm.where(labels_norm.isin(CANONICAL), other="neutral") 
    texts = df[text_col].tolist()
    y = y_true.tolist()

    results: Dict[str, Dict[str, Any]] = {}

    for analyzer in ALL_ANALYZER_NAMES:
        t0 = time.perf_counter()

        try:
            scored_df = add_sentiment_to_df(df, analyzer)
        except RuntimeError as e:
            results[analyzer] = {
                "available": False,
                "reason": str(e),
            }
            continue
        scored_df_with_tri = scored_df["sentiment"].map(_to_trinary_from_score)
        scored_df_with_tri.to_csv(f"data/benchmark/{analyzer}.csv")
        y_hat = scored_df_with_tri.tolist()
        t1 = time.perf_counter()

        m = metrics(y, y_hat)
        m["examples"] = _examples(y, y_hat, texts)
        m["time_sec"] = t1 - t0
        m["n_texts"] = len(texts)
        m["throughput_txt_per_s"] = (len(texts) / (t1 - t0)) if (t1 - t0) > 0 else float("inf")
        
        results[analyzer] = m
        
    return results


@st.cache_data(show_spinner="Running bechmark...", ttl=3600)
def run_fixed_benchmark():
    df_lab = pd.read_csv("data/benchmark/benchmark_labeled.csv")
    res = evaluate(df_lab, text_col="text", label_col="label")
    tbl = to_table(res)
    return res, tbl
