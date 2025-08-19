import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from src.utils.cache import load_cached_csv, cache_csv
from src.utils.helpers import load_csv, save_csv
import logging
import torch
import nltk
import os

logger = logging.getLogger(__name__)


_roberta = None

_FINBERT_MODEL = None
_FINBERT_TOKENIZER = None
_HF_DEVICE = int(os.environ.get("HF_DEVICE", "-1"))

_VADER = None


def vader_analyze(text: str) -> float:
    global _VADER
    if _VADER is None:
        try:
            _VADER = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            _VADER = SentimentIntensityAnalyzer()
    s = "" if text is None else str(text)
    return _VADER.polarity_scores(str(text))["compound"]    

def textblob_analyze(text:str) -> float:
    return TextBlob(str(text)).sentiment.polarity

def roberta_analyze(text: str) -> float:
    global _roberta
    if _roberta is None:
        _roberta = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
            truncation=True,
            max_length = 512,
            padding=True)
        
    short_text = str(text)[:1000]
    result = _roberta(short_text)[0]
    label = result["label"].lower()
    score = result["score"]
    return -score if label == "negative" else (score if label == "positive" else 0.0)

def _get_finbert_raw():
    global _FINBERT_MODEL,_FINBERT_TOKENIZER
    if _FINBERT_MODEL is None:
        _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        _FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        if _HF_DEVICE >= 0 and torch.cuda.is_available():
            _FINBERT_MODEL = _FINBERT_MODEL.to(f"cuda:{_HF_DEVICE}")
    return _FINBERT_MODEL, _FINBERT_TOKENIZER

def finbert_analyze(text: str) -> float:
    model, tok = _get_finbert_raw()
    short = str(text)[:1000]
    enc = tok(short, truncation=True, padding=True, max_length=512, return_tensors="pt")

    if _HF_DEVICE >= 0 and torch.cuda.is_available():
        enc = {k: v.to(f"cuda:{_HF_DEVICE}") for k, v in enc.items()}
    
    with torch.no_grad():
        logits = model(**enc).logits[0]
    
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    id2label = model.config.id2label
    pdict = {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
    
    return pdict.get("positive", 0.0) - pdict.get("negative", 0.0)

ANALYZER_UI_TO_FUNCTION = {
    "vader": vader_analyze,
    "textblob": textblob_analyze,
    "twitter-roberta": roberta_analyze,
    "finbert": finbert_analyze,
    "all": [vader_analyze, textblob_analyze, roberta_analyze, finbert_analyze]
}
ANALYZER_UI_LABELS = list(ANALYZER_UI_TO_FUNCTION.keys())

def add_sentiment_to_file(input_csv, output_csv, analyzer_name: str = "vader", cache_settings = None, freshness_minutes: int = 30):
    
    if cache_settings:
        cached = load_cached_csv(cache_settings, freshness_minutes=freshness_minutes)
        if cached is not None:
            save_csv(cached, output_csv)
            logger.info(f"Loaded cached sentiment to {output_csv}")
            return    
    
    df = load_csv(input_csv)

    analyzer_func = ANALYZER_UI_TO_FUNCTION.get(analyzer_name.lower())
    if analyzer_func is None:
        logger.error(f"Unknown analyzer: {analyzer_name}")
        raise ValueError(f"Unknown analyzer: {analyzer_name}")
    
    if isinstance(analyzer_func, list):
        for func in analyzer_func:
            col_name = f"sentiment_{func.__name__.replace('_analyze','')}"
            df[col_name] = df["text"].apply(func)
        sentiment_cols = [f"sentiment_{func.__name__.replace('_analyze','')}" for func in analyzer_func]
        df["sentiment"] = df[sentiment_cols].mean(axis=1)
        
    else:
        df["sentiment"] = df["text"].apply(analyzer_func)
    save_csv(df, output_csv)
    if cache_settings:
        cache_csv(df, cache_settings)
    logger.info(f"Sentiment added using {analyzer_name}. Saved to {output_csv}. Total records: {len(df)}")
    print("Sentiment added. Preview:")
    print(df.head())

def load_sentiment_df(reddit_path, twitter_path, posts_choice):
    if posts_choice == "Reddit":
        return pd.read_csv(reddit_path, parse_dates=["timestamp"])
    if posts_choice in ("Twitter", "Twitter/X"):
        return pd.read_csv(twitter_path, parse_dates=["timestamp"])

    r = pd.read_csv(reddit_path, parse_dates=["timestamp"]) if reddit_path else None
    t = pd.read_csv(twitter_path, parse_dates=["timestamp"]) if twitter_path else None

    frames = [df for df in (r, t) if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=["timestamp", "text", "sentiment", "source"])
    
    all_cols = sorted(set().union(*[f.columns for f in frames]))
    frames = [f.reindex(columns=all_cols) for f in frames]
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    add_sentiment_to_file("data/bitcoin_posts.csv","data/bitcoin_posts_with_sentiment.csv")