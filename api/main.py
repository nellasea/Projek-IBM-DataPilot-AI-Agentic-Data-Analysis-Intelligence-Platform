"""
api/main.py
===========
FastAPI application factory.

Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import ingest, profile, query, history
from api.schemas import HealthResponse
from core.config import get_settings
from core.logging import setup_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(
        log_level=settings.log_level.value,
        json_logs=not settings.is_development,
    )
    settings.ensure_upload_dir()
    logger.info(
        "datapilot_api_startup",
        env=settings.app_env.value,
        provider=settings.llm_provider.value,
    )
    yield
    logger.info("datapilot_api_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="DataPilot AI",
        description="Agentic Data Analysis & Intelligence Platform",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — tighten in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(ingest.router)
    app.include_router(profile.router)
    app.include_router(query.router)
    app.include_router(history.router)

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version="0.1.0",
            environment=settings.app_env.value,
        )

    return app


app = create_app()
