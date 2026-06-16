from fastapi import APIRouter, Depends, Query
import pandas as pd

from src.presentation.api.helpers.format import dataframe_to_response_models
from src.presentation.api.helpers.prep_config import posts_to_config
from src.presentation.api.schemas.posts import PostResponse
from src.presentation.api.helpers.validate import DateRangeParams
from src.app.dto import Source
from src.infra.storage.db.content_repository import load_content_df

router = APIRouter()


@router.get("/posts", response_model=list[PostResponse])
def get_posts(
    params: DateRangeParams = Depends(),
    sources: list[Source] = Query(...),
    num_posts: int = Query(10, ge=1, le=1000),
) -> list[PostResponse]:
    config = posts_to_config(params, sources, num_posts)

    posts = [load_content_df(config, source) for source in sources]
    posts_df = pd.concat(posts, ignore_index=True) if posts else pd.DataFrame()

    return dataframe_to_response_models(posts_df, PostResponse)
