from __future__ import annotations
import time
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers.pipelines import Pipeline

from textblob import TextBlob
from transformers import pipeline

from .benchmark_plot import to_table

CANONICAL = ("negative", "neutral", "positive")

def get_vader_analyzer():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # type: ignore #not worth the effort to make stub
        return SentimentIntensityAnalyzer()
    except Exception:
        import nltk # type: ignore #not worth the effort to make stub
        from nltk.sentiment import SentimentIntensityAnalyzer   # type: ignore #not worth the effort to make stub
        try:
            return SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            return SentimentIntensityAnalyzer()


def _normalize_label(s: str) -> str:
    """Map common label variants into the canonical 3-class set."""
    s = (s or "").strip().lower()
    mapping = {
        "pos": "positive", "neg": "negative", "neu": "neutral",
        "bullish": "positive", "bearish": "negative",
        "positive": "positive", "negative": "negative", "neutral": "neutral",
    }
    return mapping.get(s, s)

def _to_trinary_from_score(x: float, pos: float = 0.05, neg: float = -0.05) -> str:
    if x >= pos: return "positive"
    if x <= neg: return "negative"
    return "neutral"



def pred_vader(texts: List[str]) -> List[str]:
    sid = get_vader_analyzer()
    scores = [sid.polarity_scores(str(t))["compound"] for t in texts]
    return [_to_trinary_from_score(s) for s in scores]

def pred_textblob(texts: List[str]) -> List[str]:
    scores = [TextBlob(str(t)).sentiment.polarity for t in texts]
    return [_to_trinary_from_score(s) for s in scores]

def pred_roberta(texts: List[str], batch_size: int = 32, device: int = -1) -> List[str]:
    clf: Pipeline = pipeline(
        task: str = "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device,
    )
    out = clf(texts, batch_size=batch_size)
    return [str(r["label"]).lower() for r in out]  # negative/neutral/positive

def pred_finbert(texts: List[str], batch_size: int = 16, device: int = -1) -> List[str]:
    clf = pipeline(
        "sentiment-analysis",
        model="yiyanghkust/finbert-tone",
        truncation=True,
        device=device,
    )
    out = clf(texts, batch_size=batch_size)
    return [str(r["label"]).lower() for r in out] 



def _metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    cm  = confusion_matrix(y_true, y_pred, labels=list(CANONICAL))
    report = classification_report(y_true, y_pred, labels=list(CANONICAL), zero_division=0, digits=3)
    return {"accuracy": acc, "f1_macro": f1m, "confusion": cm, "report": report}

def _examples(y_true, y_pred, texts, k=4):
    bad = [(t, yt, yp) for t, yt, yp in zip(texts, y_true, y_pred) if yt != yp]
    return bad[:k]



def evaluate(df: pd.DataFrame,
             text_col: str = "text",
             label_col: str = "label",
             device: int = -1) -> Dict[str, Dict[str, Any]]:
    # schema check (clear error if CSV is wrong)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in labeled CSV: {sorted(missing)}")

    df = df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str)

    labels_norm = df[label_col].map(_normalize_label)
    y_true = labels_norm.where(labels_norm.isin(CANONICAL), other="neutral") 
    texts = df[text_col].tolist()
    y = y_true.tolist()

    results: Dict[str, Dict[str, Any]] = {}

    def run(fn):
        t0 = time.perf_counter()
        y_hat = fn(texts)
        t1 = time.perf_counter()
        m = _metrics(y, y_hat)
        m["examples"] = _examples(y, y_hat, texts)
        m["time_sec"] = t1 - t0
        m["n_texts"] = len(texts)
        m["throughput_txt_per_s"] = (len(texts) / (t1 - t0)) if (t1 - t0) > 0 else float("inf")
        return m

    results["VADER"]    = run(pred_vader)
    results["TextBlob"] = run(pred_textblob)
    results["RoBERTa"]  = run(lambda T=texts: pred_roberta(T, device=device))
    results["FinBERT"]  = run(lambda T=texts: pred_finbert(T, device=device))
    return results



@st.cache_data(show_spinner="Running bechmark...", ttl=3600)
def run_fixed_benchmark():
    df_lab = pd.read_csv("data/benchmark_labeled.csv")
    res = evaluate(df_lab, text_col="text", label_col="label", device=-1)
    tbl = to_table(res)
    return res, tbl