from fastapi import APIRouter, Depends

from src.app.use_cases.run_ingest import run_ingest
from src.presentation.api.helpers.prep_config import sentiment_to_config
from src.presentation.api.schemas.ingest import IngestRequest, IngestResponse
from src.presentation.api.helpers.auth import require_admin_api_key


router = APIRouter()


@router.post("/ingest",
             response_model=IngestResponse,
             dependencies=[Depends(require_admin_api_key)]
             )
def ingest(request: IngestRequest) -> IngestResponse:
    config = sentiment_to_config(
        params=request.params,
        sources=request.sources,
        num_posts=request.num_posts,
        analyzer=request.analyzer,
    )

    result = run_ingest(config)
    return IngestResponse(
        status=result.status,
        coin=result.coin,
        sources=result.sources,
        price_points=len(result.price_df),
        posts_ingested=len(result.posts_df),
        sentiment_rows=len(result.sentiment_df),
    )