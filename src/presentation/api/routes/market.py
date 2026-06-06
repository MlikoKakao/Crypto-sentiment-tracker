from fastapi import APIRouter, HTTPException
from datetime import datetime
from dataclasses import replace
from typing import Any, cast

from src.app.defaults import DEFAULT_CONFIG
from src.infra.storage.db.price_repository import load_price_df
from src.shared.helpers import is_date_correct, normalize_coin
from src.shared.dataframe_utils import format_timestamp_for_api
from src.domain.market.dto import IndicatorConfig

router = APIRouter()


@router.get("/prices")
def get_prices(
    coin: str, start_date: datetime, end_date: datetime
) -> list[dict[str, Any]]:
    if not is_date_correct(start_date, end_date):
        raise HTTPException(status_code=400, detail="end_date must be after start_date")
    coin = normalize_coin(coin)

    config = replace(
        DEFAULT_CONFIG,
        coin=coin,
        start_date=start_date,
        end_date=end_date,
    )
    df = load_price_df(config)
    df = format_timestamp_for_api(df)
    return cast(list[dict[str, Any]], df.to_dict(orient="records"))


@router.get("/signals")
def get_signals(
    coin: str,
    start_date: datetime,
    end_date: datetime,
    use_sma: bool = False,
    use_rsi: bool = False,
    use_macd: bool = False,
) -> list[dict[str, Any]]:
    if not is_date_correct(start_date, end_date):
        raise HTTPException(status_code=400, detail="end_date must be after start_date")
    if not any([use_sma, use_rsi, use_macd]):
        raise HTTPException(
            status_code=400, detail="At least one signal must be selected"
        )
    try:
        coin = normalize_coin(coin)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    config = replace(
        DEFAULT_CONFIG,
        coin=coin,
        start_date=start_date,
        end_date=end_date,
    )

    request = IndicatorConfig(
        coin=coin,
        start_date=start_date,
        end_date=end_date,
        use_sma=use_sma,
        use_rsi=use_rsi,
        use_macd=use_macd,
    )

    price_df = load_price_df(config)

    from src.app.use_cases.get_indicators import add_indicators_with_cache

    signals_df = add_indicators_with_cache(price_df, request)

    signals_df = format_timestamp_for_api(signals_df)
    return cast(list[dict[str, Any]], signals_df.to_dict(orient="records"))
