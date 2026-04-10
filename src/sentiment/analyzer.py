import pandas as pd
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore[import]
except Exception:  # pragma: no cover - fallback when nltk not installed
    SentimentIntensityAnalyzer = None  # type: ignore

try:
    from textblob import TextBlob  # type: ignore[import]
except Exception:  # pragma: no cover - fallback when textblob not installed
    TextBlob = None  # type: ignore

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification  # type: ignore[import]
except Exception:  # pragma: no cover - fallback when transformers not installed
    pipeline = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore

try:
    import torch  # type: ignore[import]
    import torch.nn.functional as F  # type: ignore[import]
except Exception:  # pragma: no cover - torch may not be installed in dev env
    torch = None  # type: ignore
    F = None  # type: ignore
from src.utils.cache import load_cached_csv, cache_csv
from src.utils.helpers import load_csv, save_csv
import logging
try:
    import nltk  # type: ignore[import]
except Exception:
    nltk = None  # type: ignore
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


_roberta: Any = None

_finbert_model: Any = None
_finbert_tokenizer: Any = None
_hf_device: int = int(os.environ.get("HF_DEVICE", "-1"))

_vader: Any = None

# Cast external classes/functions to Any aliases so the static checker
# doesn't complain about possible None types after guarded imports.
SentimentIntensityAnalyzer_cls: Any = SentimentIntensityAnalyzer  # type: ignore
TextBlob_cls: Any = TextBlob  # type: ignore
pipeline_fn: Any = pipeline  # type: ignore
AutoTokenizer_cls: Any = AutoTokenizer  # type: ignore
AutoModelForSequenceClassification_cls: Any = AutoModelForSequenceClassification  # type: ignore
torch_mod: Any = torch  # type: ignore
F_mod: Any = F  # type: ignore


def vader_analyze(text: Optional[str]) -> float:
    global _vader
    if _vader is None:
        if SentimentIntensityAnalyzer_cls is None:
            raise RuntimeError("nltk SentimentIntensityAnalyzer not available")
        try:
            _vader = SentimentIntensityAnalyzer_cls()
        except LookupError:
            if nltk is not None:
                try:
                    nltk.download("vader_lexicon", quiet=True)
                except Exception:
                    pass
            _vader = SentimentIntensityAnalyzer_cls()
    s = "" if text is None else str(text)
    return _vader.polarity_scores(s)["compound"]

def textblob_analyze(text: Optional[str]) -> float:
    if TextBlob_cls is None:
        raise RuntimeError("textblob not available")
    return float(getattr(TextBlob_cls(str(text)).sentiment, "polarity", 0.0))

def roberta_analyze(text: Optional[str]) -> float:
    global _roberta
    if _roberta is None:
        if pipeline_fn is None:
            raise RuntimeError("transformers.pipeline not available")
        _roberta = pipeline_fn(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
            truncation=True,
            max_length=512,
            padding=True,
        )
        
    short_text = str(text)[:1000]
    result = _roberta(short_text)[0]
    label = result["label"].lower()
    score = result["score"]
    return -score if label == "negative" else (score if label == "positive" else 0.0)

def _get_finbert_raw() -> tuple[Any, Any]:
    global _finbert_model, _finbert_tokenizer
    if _finbert_model is None or _finbert_tokenizer is None:
        if AutoTokenizer_cls is None or AutoModelForSequenceClassification_cls is None:
            return None, None
        _finbert_tokenizer = AutoTokenizer_cls.from_pretrained("yiyanghkust/finbert-tone")
        _finbert_model = AutoModelForSequenceClassification_cls.from_pretrained("yiyanghkust/finbert-tone")
        if _hf_device >= 0 and torch_mod is not None and getattr(torch_mod, "cuda", None) is not None and torch_mod.cuda.is_available():
            _finbert_model = _finbert_model.to(f"cuda:{_hf_device}")
    return _finbert_model, _finbert_tokenizer

def finbert_analyze(text: str) -> float:
    model, tok = _get_finbert_raw()
    if model is None or tok is None:
        raise RuntimeError("FinBERT model/tokenizer not available")
    short = str(text)[:1000]
    enc = tok(short, truncation=True, padding=True, max_length=512, return_tensors="pt")

    if _hf_device >= 0 and torch_mod is not None and getattr(torch_mod, "cuda", None) is not None and torch_mod.cuda.is_available():
        enc = {k: v.to(f"cuda:{_hf_device}") for k, v in enc.items()}

    with torch_mod.no_grad():
        logits = model(**enc).logits[0]

    probs = F_mod.softmax(logits, dim=-1).detach().cpu().numpy()
    id2label = model.config.id2label
    pdict = {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
    
    return pdict.get("positive", 0.0) - pdict.get("negative", 0.0)

ANALYZER_UI_TO_FUNCTION: Dict[str, Any] = {
    "vader": vader_analyze,
    "textblob": textblob_analyze,
    "twitter-roberta": roberta_analyze,
    "finbert": finbert_analyze,
    "all": [vader_analyze, textblob_analyze, roberta_analyze, finbert_analyze],
}
ANALYZER_UI_LABELS = list(ANALYZER_UI_TO_FUNCTION.keys())

def add_sentiment_to_file(
    input_csv: str,
    output_csv: str,
    analyzer_name: str = "vader",
    cache_settings: Optional[Dict[str, Any]] = None,
    freshness_minutes: int = 30,
) -> None:

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
            col_name = f"sentiment_{func.__name__.replace('_analyze', '')}"
            df[col_name] = df["text"].apply(func)
        sentiment_cols = [f"sentiment_{func.__name__.replace('_analyze', '')}" for func in analyzer_func]
        df["sentiment"] = df[sentiment_cols].mean(axis=1)

    else:
        df["sentiment"] = df["text"].apply(analyzer_func)
    save_csv(df, output_csv)
    if cache_settings:
        cache_csv(df, cache_settings)
    logger.info(
        f"Sentiment added using {analyzer_name}. Saved to {output_csv}. Total records: {len(df)}"
    )
    print("Sentiment added. Preview:")
    print(df.head())

def load_sentiment_df(
    news_path: Optional[str], reddit_path: Optional[str], twitter_path: Optional[str], posts_choice: str
) -> pd.DataFrame:
    if posts_choice == "News":
        if not news_path:
            return pd.DataFrame(columns=["timestamp", "text", "sentiment", "source"])
        return pd.read_csv(news_path, parse_dates=["timestamp"]).copy()
    if posts_choice == "Reddit":
        if not reddit_path:
            return pd.DataFrame(columns=["timestamp", "text", "sentiment", "source"])
        return pd.read_csv(reddit_path, parse_dates=["timestamp"]).copy()
    if posts_choice in ("Twitter", "Twitter/X"):
        if not twitter_path:
            return pd.DataFrame(columns=["timestamp", "text", "sentiment", "source"])
        return pd.read_csv(twitter_path, parse_dates=["timestamp"]).copy()

    n = pd.read_csv(news_path, parse_dates=["timestamp"]) if news_path else None
    r = pd.read_csv(reddit_path, parse_dates=["timestamp"]) if reddit_path else None
    t = pd.read_csv(twitter_path, parse_dates=["timestamp"]) if twitter_path else None

    frames = [df for df in (n, r, t) if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=["timestamp", "text", "sentiment", "source"])
    
    all_cols = sorted(set().union(*[f.columns for f in frames]))
    frames = [f.reindex(columns=all_cols) for f in frames]
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    add_sentiment_to_file("data/bitcoin_posts.csv","data/bitcoin_posts_with_sentiment.csv")