import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline
from src.utils.helpers import load_csv, save_csv
import logging
import nltk
nltk.download("vader_lexicon", quiet=True)
logger = logging.getLogger(__name__)



def vader_analyze(text: str) -> float:
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(str(text))["compound"]       

def textblob_analyze(text:str) -> float:
    return TextBlob(str(text)).sentiment.polarity

roberta_pipeline = pipeline("sentiment-analysis",
                            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                            truncation=True,
                            max_length = 512,
                            padding=True)

def roberta_analyze(text: str) -> float:
    short_text = str(text)[:1000]
    result = roberta_pipeline(short_text)[0]
    label = result["label"].lower()
    score = result["score"]
    if label == "negative":
        return -score
    elif label == "positive":
        return score
    else:
        return 0

ANALYZER_UI_TO_FUNCTION = {
    "vader": vader_analyze,
    "textblob": textblob_analyze,
    "twitter-roberta": roberta_analyze
}
ANALYZER_UI_LABELS = list(ANALYZER_UI_TO_FUNCTION.keys())


def add_sentiment_to_file(input_csv, output_csv, analyzer_name: str = "vader"):
    df = load_csv(input_csv)
    analyzer_func = ANALYZER_UI_TO_FUNCTION.get(analyzer_name.lower())
    if analyzer_func is None:
        logger.error(f"Unknown analyzer: {analyzer_name}")
        raise ValueError(f"Unknown analyzer: {analyzer_name}")
    df["sentiment"] = df["text"].apply(analyzer_func)
    save_csv(df, output_csv)
    logging.info(f"Sentiment added using {ANALYZER_UI_LABELS}. Saved to {output_csv}. Total records: {len(df)}")
    print("Sentiment added. Preview:")
    print(df.head())

if __name__ == "__main__":
    add_sentiment_to_file("data/bitcoin_posts.csv","data/bitcoin_posts_with_sentiment.csv")