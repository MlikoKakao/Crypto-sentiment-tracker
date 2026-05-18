from fastapi import FastAPI
from src.infra.fetchers.coinbase_price import get_coinbase_price_history
from src.app.defaults import DEFAULT_CONFIG
from dataclasses import replace
from datetime import datetime
from src.domain.sentiment.service import add_sentiment_to_df
from src.infra.fetchers.service import fetch_posts
from src.domain.market.indicators import add_indicators_to_df
from src.presentation.sidebar import IndicatorConfig

app = FastAPI()


@app.get("/ping")
def pong():
    return {"ping": "pong!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/prices")
def get_prices(
    coin: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
):
    if not (coin and start_date and end_date):
        config = DEFAULT_CONFIG
    else:
        config = replace(
            DEFAULT_CONFIG, coin=coin.upper(), start_date=start_date, end_date=end_date
        )
    return get_coinbase_price_history(config).to_dict(orient="records")


@app.get("/sentiment")
def get_sentiment(
    coin: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    analyzer: str | None = None,
    sources: str | None = None,
):
    if not (coin and start_date and end_date and analyzer and sources):
        config = DEFAULT_CONFIG
    else:
        config = replace(
            DEFAULT_CONFIG,
            coin=coin.upper(),
            start_date=start_date,
            end_date=end_date,
            analyzer=analyzer,
            sources=sources,
        )
    df = fetch_posts(config)
    sentiment_df = add_sentiment_to_df(df) 
    return sentiment_df.to_dict(orient="records")


# Signals take in state .. so idk what to do, is function overloading in python?
@app.get("/signals")
def get_signals(
    coin: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    use_sma: bool = True,
    use_rsi: bool = True,
    use_macd: bool = True,
    sma_fast: int = 20,
    sma_slow: int = 50,
    rsi_period: int = 14,
):
    if not (coin and start_date and end_date):
        config = DEFAULT_CONFIG
    else:
        config = replace(
            DEFAULT_CONFIG, coin=coin.upper(), start_date=start_date, end_date=end_date
        )

    indicator_config = IndicatorConfig(
        coin=config.coin,
        use_sma=use_sma,
        use_rsi=use_rsi,
        use_macd=use_macd,
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        rsi_period=rsi_period,
        start_date=config.start_date,
        end_date=config.end_date,
    )

    price_df = get_coinbase_price_history(config)
    signals_df = add_indicators_to_df(price_df, indicator_config)

    return signals_df.to_dict(orient="records")