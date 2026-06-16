from fastapi import APIRouter, Depends

from src.infra.storage.db.price_repository import load_price_df
from src.presentation.api.helpers.format import dataframe_to_response_models
from src.presentation.api.helpers.prep_config import (
    build_indicator_config,
    date_range_to_config,
)
from src.presentation.api.helpers.validate import DateRangeParams
from src.presentation.api.schemas.prices import PricePoint
from src.presentation.api.schemas.signals import SignalResponse

router = APIRouter()


@router.get("/prices", response_model=list[PricePoint])
def get_prices(params: DateRangeParams = Depends()) -> list[PricePoint]:
    config = date_range_to_config(params)
    df = load_price_df(config)

    return dataframe_to_response_models(df, PricePoint)


@router.get("/signals", response_model=list[SignalResponse])
def get_signals(
    params: DateRangeParams = Depends(),
    use_sma: bool = False,
    use_rsi: bool = False,
    use_macd: bool = False,
) -> list[SignalResponse]:
    config = date_range_to_config(params)
    request = build_indicator_config(params, use_sma, use_rsi, use_macd)

    price_df = load_price_df(config)

    from src.app.use_cases.get_indicators import add_indicators_with_cache

    signals_df = add_indicators_with_cache(price_df, request)

    return dataframe_to_response_models(signals_df, SignalResponse)
