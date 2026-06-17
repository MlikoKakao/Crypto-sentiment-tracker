from fastapi import APIRouter

from src.app.use_cases.run_ingest import run_ingest
from src.presentation.api.helpers.prep_config import sentiment_to_config
from src.presentation.api.schemas.ingest import IngestRequest, IngestResponse


router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    config = sentiment_to_config(
        params=request.params,
        sources=request.sources,
        num_posts=request.num_posts,
        analyzer=request.analyzer,
    )

    return IngestResponse.model_validate(run_ingest(config))
