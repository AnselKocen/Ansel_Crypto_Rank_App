"""
Ansel Crypto Rank App - FastAPI Web Application
Main entry point for the web server
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add parent directory to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Load .env file
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _k, _v = _k.strip(), _v.strip()
            # Chat settings are read on demand so saving a key/model needs no restart.
            if _k in {"MOONSHOT_API_KEY", "KIMI_MODEL", "KIMI_BASE_URL", "KIMI_REASONING_EFFORT"}:
                continue
            if _v and not os.environ.get(_k):
                os.environ[_k] = _v

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import uvicorn

# Import routers
from web.routers import pages
from web.routers import api_sentiment
from web.routers import api_analysis
from web.routers import api_chat
from web.routers import api_predictions
from web.routers import api_skills
from web.routers import api_figures
from web.routers import api_game
from web.routers import api_replay
from web.routers import api_updates
from web.services.weekly_updates import get_updates
from web.services.market_catalog import CatalogRefresher


@asynccontextmanager
async def lifespan(app):
    service = get_updates()
    catalog = CatalogRefresher()
    if os.environ.get("ANSEL_AUTO_UPDATE", "1") != "0":
        service.start()
        catalog.start()
    yield
    service.close()
    catalog.close()

# Create FastAPI application
app = FastAPI(
    title="Ansel Crypto Rank App",
    description="Machine Learning-driven Cryptocurrency Recommendation System",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "web" / "static"), name="static")
app.mount("/figures", StaticFiles(directory=BASE_DIR / "figures"), name="figures")

# Setup templates
templates = Jinja2Templates(directory=BASE_DIR / "web" / "templates")

# Include routers
app.include_router(pages.router)
app.include_router(api_sentiment.router, prefix="/api")
app.include_router(api_analysis.router, prefix="/api")
app.include_router(api_chat.router, prefix="/api")
app.include_router(api_predictions.router, prefix="/api")
app.include_router(api_skills.router, prefix="/api")
app.include_router(api_figures.router, prefix="/api")
app.include_router(api_game.router, prefix="/api")
app.include_router(api_replay.router, prefix="/api")
app.include_router(api_updates.router, prefix="/api")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0"}


if __name__ == "__main__":
    uvicorn.run(
        "web.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
