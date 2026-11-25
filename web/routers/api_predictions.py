"""
Predictions API Routes
"""

import sys
import json
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pathlib import Path
from web.services.prediction_service import (
    get_available_models,
    load_predictions
)
from web.services.paper_trading_service import get_paper_trading_data

router = APIRouter(prefix="/predictions", tags=["predictions"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEXT_DIR = BASE_DIR / "text"
sys.path.insert(0, str(BASE_DIR))


@router.get("/models")
async def get_models():
    """Get available prediction models"""
    models = get_available_models()
    return {"models": models}


@router.get("/data/{model}")
async def get_prediction_data(model: str):
    """Get predictions for a specific model"""
    predictions = load_predictions(model)
    if predictions is None:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found or no predictions available")
    return predictions


@router.get("/text/{filename}")
async def get_text_file(filename: str):
    """Get content of a text file from the text directory"""
    # Security: only allow specific files
    allowed_files = ["strategy_intro.txt", "best_params.txt", "enet_features.txt"]
    if filename not in allowed_files:
        raise HTTPException(status_code=403, detail="File not allowed")

    file_path = TEXT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

    content = file_path.read_text(encoding='utf-8')
    return {"filename": filename, "content": content}


@router.get("/paper-trading")
async def get_paper_trading(capital: float = 10000.0):
    """
    Get paper trading simulation data.
    Compares extra_ls (Long-Short) vs extra_EW (Equal Weight) strategies.
    """
    try:
        data = get_paper_trading_data(initial_capital=capital)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Paper trading calculation failed: {str(e)}")


async def run_prediction_stream():
    """Compatibility stream backed by the shared job; never starts a script."""
    from web.services.weekly_updates import get_updates
    service = get_updates()
    request = service.request("prediction", "data")
    yield "data: " + json.dumps({"type": "status", **request}) + "\n\n"
    while service.status("data")["running"]:
        await asyncio.sleep(2)
        yield "data: " + json.dumps({"type": "status", **service.status("data")}) + "\n\n"
    yield "data: " + json.dumps({"type": "done", **service.status("data")}) + "\n\n"


@router.post("/run")
async def run_prediction():
    return StreamingResponse(run_prediction_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})
