import subprocess

from src.scraping import fetch_reddit_posts, get_price_history
from src.sentiment import add_sentiment_to_file
from src.processing.merge_data import merge_sentiment_and_price

class CryptoSentimentApp:
    def __init__(self):
        self.posts_file = "data/bitcoin_posts.csv"
        self.sentiment_file = "data/bitcoin_posts_with_sentiment.csv"
        self.price_file = "data/bitcoin_prices.csv"
        self.merged_file = "data/merged_sentiment_price.csv"

    def run(self):
        print("Starting!")
        self.fetch_reddit_posts()
        self.analyze_sentiment()
        self.fetch_price_data()
        self.merge_data()
        self.launch_dashboard()

    def fetch_reddit_posts(self):
        print("Fetching Reddit posts")
        df = fetch_reddit_posts("bitcoin", limit=1000)
        df.to_csv(self.posts_file, index=False)
        print(f"Saved posts to {self.posts_file}")

    def analyze_sentiment(self):
        print("Running sentiment analysis")
        add_sentiment_to_file(self.posts_file, self.sentiment_file)
        print(f"Saved sentiment to {self.sentiment_file}")

    def fetch_price_data(self):
        print("Fetching BTC price")
        df = get_price_history("bitcoin",days="365")
        df.to_csv(self.price_file, index=False)
        print(f"Saved to {self.price_file}")

    def merge_data(self):
        print("Merging data")
        merge_sentiment_and_price(self.sentiment_file, self.price_file, self.merged_file)
        print(f"Merged to {self.merged_file}")

    def launch_dashboard(self):
        print("Launching dashboard")
        subprocess.run(["streamlit","run","src/dashboard.py"])
    
if __name__ == "__main__":
    app = CryptoSentimentApp()
    app.run()