import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    return score["compound"]        

def add_sentiment_to_file(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df["sentiment"] = df["text"].apply(analyze_sentiment)
    df.to_csv(output_csv, index=False)
    print("Sentiment added. Preview:")
    print(df.head())

if __name__ == "__main__":
    add_sentiment_to_file("data/bitcoin_posts.csv","data/bitcoin_posts_with_sentiment.csv")