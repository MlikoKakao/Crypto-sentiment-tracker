from fastapi import FastAPI

from src.presentation.api.routes.health import router as health_router
from src.presentation.api.routes.market import router as market_router
from src.presentation.api.routes.posts import router as posts_router
from src.presentation.api.routes.sentiment import router as sentiment_router
from src.presentation.api.routes.ingest import router as ingest_router

app = FastAPI()

app.include_router(health_router)
app.include_router(market_router, prefix="/market", tags=["market"])
app.include_router(posts_router)
app.include_router(sentiment_router)
app.include_router(ingest_router)
