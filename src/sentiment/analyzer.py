import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline
from src.utils.cache import load_cached_csv, cache_csv
from src.utils.helpers import load_csv, save_csv
import logging
import nltk
nltk.download("vader_lexicon", quiet=True)
logger = logging.getLogger(__name__)
_roberta = None



def vader_analyze(text: str) -> float:
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(str(text))["compound"]       

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
    return -score if label == "negative" else (score if label == "positive" else 0)

ANALYZER_UI_TO_FUNCTION = {
    "vader": vader_analyze,
    "textblob": textblob_analyze,
    "twitter-roberta": roberta_analyze,
    "all": [vader_analyze, textblob_analyze, roberta_analyze]
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

if __name__ == "__main__":
    add_sentiment_to_file("data/bitcoin_posts.csv","data/bitcoin_posts_with_sentiment.csv")