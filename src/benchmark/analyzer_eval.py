from __future__ import annotations
import time
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline

CANONICAL = ("negative", "neutral", "positive")


# --- helpers ---------------------------------------------------------------

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
    sid = SentimentIntensityAnalyzer()
    scores = [sid.polarity_scores(str(t))["compound"] for t in texts]
    return [_to_trinary_from_score(s) for s in scores]

def pred_textblob(texts: List[str]) -> List[str]:
    scores = [TextBlob(str(t)).sentiment.polarity for t in texts]
    return [_to_trinary_from_score(s) for s in scores]

def pred_roberta(texts: List[str], batch_size: int = 32, device: int = -1) -> List[str]:
    clf = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
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

def to_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "Accuracy": r["accuracy"],
            "F1 (macro)": r["f1_macro"],
            "Throughput (texts/s)": r.get("throughput_txt_per_s", np.nan),
            "Latency (ms/text)": (1000 * r["time_sec"] / max(r.get("n_texts", 1), 1)) if r.get("time_sec") else np.nan,
        })
    df = (pd.DataFrame(rows)
            .sort_values(["F1 (macro)", "Accuracy"], ascending=False)
            .reset_index(drop=True))
    df.index = np.arange(1, len(df) +1)
    df.index.name = "Rank"
    return df

def confusion_figure(cm: np.ndarray, labels=CANONICAL, title: str = "Confusion matrix"):
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    fig = px.imshow(
        df_cm, text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True", color="Count"),
        title=title, aspect="equal"
    )
    return fig
