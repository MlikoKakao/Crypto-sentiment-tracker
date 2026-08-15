from dataclasses import replace

from src.app.defaults import default_config
from src.app.dto import AnalysisConfig, Analyzer, Source
from src.domain.market.dto import IndicatorConfig
from src.presentation.api.helpers.validate import DateRangeParams
from src.shared.helpers import normalize_coin


def date_range_to_config(params: DateRangeParams) -> AnalysisConfig:
    return replace(
        default_config(),
        coin=normalize_coin(params.coin),
        start_date=params.start_date,
        end_date=params.end_date,
    )


def build_indicator_config(
    params: DateRangeParams,
    use_sma: bool,
    use_rsi: bool,
    use_macd: bool,
) -> IndicatorConfig:
    return IndicatorConfig(
        coin=normalize_coin(params.coin),
        start_date=params.start_date,
        end_date=params.end_date,
        use_sma=use_sma,
        use_rsi=use_rsi,
        use_macd=use_macd,
    )


def posts_to_config(
    params: DateRangeParams,
    sources: list[Source],
    num_posts: int,
) -> AnalysisConfig:
    return replace(
        date_range_to_config(params),
        sources=tuple(sources),
        num_posts=num_posts,
    )


def sentiment_to_config(
    params: DateRangeParams,
    sources: list[Source],
    num_posts: int,
    analyzer: Analyzer,
) -> AnalysisConfig:
    return replace(
        posts_to_config(params, sources, num_posts),
        analyzer=analyzer,
    )
